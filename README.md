# Breaking Multi-Agent Stasis

核心代码：自适应双层金融多智能体量化交易框架。

## 框架结构

- **组织层**：根据市场状态、可用资源和服务可靠性选择能力子集，约束令牌预算、延迟和联合故障概率，并支持失败注入、退出/重入、拓扑切换和风险平价锚点。
- **决策层**：将趋势、均值回复、风险平价和横截面等能力的目标组合起来，再投影到基准偏离、跟踪误差、换手和单资产权重约束内。

核心入口是 `src/tradingagents/portfolio/emergent_organization_framework.py`。


## 安装

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\Activate.ps1
python -m pip install -e ".[dev]"
```

## 最小示例

```python
import numpy as np
from tradingagents.portfolio.emergent_organization_framework import (
    BudgetedOrganizationController, OrganizationConfig,
)

controller = BudgetedOrganizationController(
    OrganizationConfig(
        token_budget=1800,
        latency_budget_ms=320,
        joint_failure_budget=0.10,
        anchor_agent="risk_parity",
        anchor_weight=0.75,
        seed=7,
    ),
    expert_names=("trend", "meanrev", "risk_parity", "cross_sectional"),
)
available = {name: True for name in controller.expert_names}
weights = controller.allocate(available, regime="down_high_vol")
print(weights)

# 只向实际活跃且成功服务的能力反馈已实现净收益。
controller.update({"trend": 0.004, "risk_parity": 0.002})
```

## 测试

```bash
pytest
```

`tests/` 覆盖硬预算、失败服务、退出/重入、延迟反馈、锚定权重和基准回退。

## 目录

```text
github_bundle/
  src/tradingagents/portfolio/  # 核心算法
  tests/                        # 单元测试
  pyproject.toml                # 可编辑安装与 pytest 配置
  requirements.txt              # 运行时依赖
```

