import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================
# 1. 基础组件：静态卷积与标准下采样 (用于光学分支)
# =============================================================
class DilatedDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=2, dilation=2),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )

    def forward(self, x): return self.conv(x)


class StaticDown(nn.Module):
    """标准卷积下采样：快速锁定空间几何结构"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            DilatedDoubleConv(out_ch, out_ch)
        )

    def forward(self, x): return self.conv(x)


# =============================================================
# 2. 液态组件：液态下采样 (专门用于 SAR 分支)
# =============================================================
class LiquidDown(nn.Module):
    """液态下采样：在下采样过程中通过动力学演化滤除相干斑噪声"""

    def __init__(self, in_ch, out_ch, steps=4, dt=0.2):
        super().__init__()
        self.steps = steps
        self.dt = dt

        # 空间压缩
        self.compress = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)

        # 演化函数 f(h)
        self.f = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
            nn.Conv2d(out_ch, out_ch, 1),
            nn.SiLU()
        )
        # 自适应时间常数
        self.tau_gen = nn.Sequential(nn.Conv2d(out_ch, 1, 1), nn.Sigmoid())

    def forward(self, x):
        h = self.bn(self.compress(x))
        input_stimulus = h.clone()
        for _ in range(self.steps):
            tau = self.tau_gen(h) + 0.1
            # LNN 方程: dh/dt = -1/tau * h + f(x)
            dh = -(1.0 / tau) * h + self.f(input_stimulus)
            h = h + self.dt * dh
            h = torch.tanh(h)
        return h


# =============================================================
# 3. 跨模态引导与演化单元 (SGS & TLC)
# =============================================================
class StructuralGuidedSuppressor(nn.Module):
    """利用光学结构引导 SAR 的特征对齐"""

    def __init__(self, channels, steps=4):
        super().__init__()
        self.steps = steps
        self.interaction = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid()
        )
        self.tau_gen = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())

    def forward(self, x_sar, f_opt):
        guide_mask = self.interaction(torch.cat([x_sar, f_opt], dim=1))
        h = x_sar.clone()
        for _ in range(self.steps):
            tau = self.tau_gen(h * guide_mask) + 0.1
            dh = -(1.0 / tau) * h + x_sar
            h = h + 0.2 * dh
            h = torch.tanh(h)
        return h


class TemporalLiquidCell(nn.Module):
    def __init__(self, channels, steps=6, dt=0.1):  # 💡 1. 把离散积分步长从 0.2 缩流到 0.1，保证离散动力系统的绝对数值稳定性！
        super().__init__()
        self.steps = steps
        self.dt = dt
        self.f_stimulus = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.Tanh()
        )
        # 💡 2. 将最后一层改为 Softplus，或者保持 Sigmoid 但注入一个缩放因子，让 tau 不会变得太小
        self.tau_gen = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )
        self.change_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, h_prev, stimulus):
        state = h_prev.clone()
        stim = self.f_stimulus(stimulus)

        for _ in range(self.steps):
            # 💡 3. 核心修正：加上一个合理的基底（比如 0.5），让 1/tau 最大不会超过 2，配合 dt=0.1，演化轨迹会极其丝滑平稳！
            tau = self.tau_gen(state + stimulus) * 1.5 + 0.1
            dh = -(1.0 / tau) * state + stim
            state = state + self.dt * dh
            state = torch.tanh(state)

        return self.change_proj(state - h_prev)


# =============================================================
# 4. 🔥 核心架构：非对称液态网络 (HRSICD)
# =============================================================
class HRSICD(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()

        # --- [1] 光学 Encoder 流 (静态卷积) ---
        self.inc_opt = DilatedDoubleConv(n_channels, 64)
        self.down1_opt = StaticDown(64, 128)
        self.down2_opt = StaticDown(128, 256)

        # --- [2] SAR Encoder 流 (液态演化) ---
        self.inc_sar = DilatedDoubleConv(n_channels, 64)
        self.down1_sar = LiquidDown(64, 128)
        self.down2_sar = LiquidDown(128, 256)

        # --- [3] 结构引导与差异挖掘 ---
        self.sgs0 = StructuralGuidedSuppressor(64)
        self.sgs1 = StructuralGuidedSuppressor(128)
        self.sgs2 = StructuralGuidedSuppressor(256)

        self.tlc0 = TemporalLiquidCell(64)
        self.tlc1 = TemporalLiquidCell(128)
        self.tlc2 = TemporalLiquidCell(256)

        # --- [4] 解码与融合路径 ---
        self.up2 = nn.Sequential(nn.ConvTranspose2d(256, 128, 2, stride=2), nn.SiLU())
        self.up1 = nn.Sequential(nn.ConvTranspose2d(128, 64, 2, stride=2), nn.SiLU())

        # --- [5] 分类头与辅助头 (兼容 main.py) ---
        self.head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, n_classes, 1)
        )
        self.aux_head2 = nn.Conv2d(128, n_classes, 1)  # 对应 1/2 尺度
        self.aux_head1 = nn.Conv2d(64, n_classes, 1)  # 对应原图尺度

    def forward(self, t1_opt, t2_sar, with_aux=False):
        # 1. 编码 (光学静态 vs SAR 液态)
        o0 = self.inc_opt(t1_opt)
        o1 = self.down1_opt(o0)
        o2 = self.down2_opt(o1)

        s0 = self.sgs0(self.inc_sar(t2_sar), o0)
        s1 = self.sgs1(self.down1_sar(s0), o1)
        s2 = self.sgs2(self.down2_sar(s1), o2)

        # 2. 液态差异挖掘
        diff2 = self.tlc2(o2, s2)  # 1/4 尺寸
        diff1 = self.tlc1(o1, s1)  # 1/2 尺寸
        diff0 = self.tlc0(o0, s0)  # 原图尺寸

        # 3. 逐级融合解码
        d2 = self.up2(diff2) + diff1
        d1 = self.up1(d2) + diff0

        # 4. 输出
        out = self.head(d1)

        if with_aux:
            # 返回: 主输出, 辅助2(1/2尺度), 辅助1(原图尺度)
            # 注意: 为了计算损失，我们需要将 aux 结果 resize 到原图大小
            a2 = F.interpolate(self.aux_head2(diff1), size=t1_opt.shape[2:], mode='bilinear')
            a1 = self.aux_head1(diff0)
            return out, a2, a1

        return out

    def reset_memory(self):
        pass


if __name__ == "__main__":
    model = HRSICD()
    t1 = torch.randn(1, 3, 64, 64)
    t2 = torch.randn(1, 3, 64, 64)  # 假设 SAR 也是 3 通道
    # 测试主输出
    out = model(t1, t1)
    # 测试辅助输出
    pred, aux2, aux1 = model(t1, t1, with_aux=True)
    print(f"✅ 架构对齐成功！")
    print(f"主输出: {pred.shape}, 辅助2: {aux2.shape}, 辅助1: {aux1.shape}")