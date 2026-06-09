# 🧠 Explainable Reinforcement Learning for Safe Batch Reactor Control

## 📖 Project Overview

This project presents an Explainable Reinforcement Learning (XRL) framework for the safe and resilient control of a laboratory-scale acrylamide polymerization batch reactor. The system employs a Proximal Policy Optimization (PPO) agent to continuously regulate reactor temperature by coordinating two control actuators: heater current and coolant flow rate.

To improve reliability in safety-critical environments, the framework integrates an Unscented Kalman Filter (UKF) for real-time state estimation and multiple Explainable AI techniques, including SHAP analysis, surrogate decision trees, and sensitivity heatmaps, to provide transparency into the agent's decision-making process.

The controller successfully maintains reactor temperature within operational limits, prevents thermal runaway, and ensures safe operation throughout a complete 60-hour batch cycle.

---

## 🎯 Objectives

* Develop a reinforcement learning controller for nonlinear chemical process control.
* Achieve precise reactor temperature tracking using continuous control actions.
* Prevent thermal runaway in a safety-critical batch reactor environment.
* Estimate unobservable reactor states using Unscented Kalman Filtering.
* Improve policy transparency through Explainable AI techniques.
* Evaluate system resilience under dynamic operating conditions.

---

## 🤖 Reinforcement Learning Framework

### PPO Agent

The control strategy is based on Proximal Policy Optimization (PPO), a state-of-the-art policy gradient reinforcement learning algorithm designed for continuous action spaces.

### State Space

The agent observes:

* Estimated Initiator Concentration
* Estimated Monomer Concentration
* Reactor Temperature
* Jacket Temperature
* Temperature Tracking Error

### Action Space

The PPO agent continuously controls:

* Heater Current (8–20 mA)
* Coolant Flow Rate (0–0.7 L/min)

---

## 🧩 Explainable AI Components

The project incorporates multiple explainability approaches:

### 🌳 Surrogate Decision Trees

Generate human-readable control rules that approximate PPO policy behavior.

### 📊 SHAP Analysis

Identify the contribution of each state variable to control decisions.

### 🔥 Sensitivity Heatmaps

Visualize how changes in reactor states influence control actions.

These methods transform the learned policy from a black-box controller into an interpretable and auditable system.

---

## 📊 Reactor Environment

The simulation is based on a laboratory-scale acrylamide polymerization reactor involving:

* Exothermic chemical reactions
* Nonlinear reactor dynamics
* Temperature-sensitive reaction kinetics
* Dual-actuator process control

The environment is modeled using Ordinary Differential Equations (ODEs) representing:

* Initiator Concentration
* Monomer Concentration
* Reactor Temperature
* Jacket Temperature

---

## 🚀 Key Features

✅ PPO-based continuous control

✅ Dual-actuator optimization

✅ Unscented Kalman Filter (UKF) state estimation

✅ Resilience-aware reward engineering

✅ Thermal runaway prevention

✅ Explainable AI integration

✅ SHAP feature attribution

✅ Decision tree policy interpretation

✅ Sensitivity analysis visualization

✅ PyTorch-based implementation

---

## ⚙️ Installation & Setup

### Clone the Repository

```bash
git clone https://github.com/yourusername/ppo-reactor-controller.git
cd ppo-reactor-controller
```

### Train PPO Agent

```bash
python train_ppo.py
```

### Run Explainability Analysis

```bash
python explainability/explain_shap.py
python explainability/explain_tree.py
python explainability/explain_sensitivity.py
```

---

## 🧠 State Variables

| Variable | Description |
|-----------|-------------|
| Î | Estimated Initiator Concentration |
| M̂ | Estimated Monomer Concentration |
| Tr | Reactor Temperature |
| Tj | Jacket Temperature |
| e | Temperature Tracking Error |

---

## 🎛️ Control Variables

| Control Action | Range |
|---------------|--------|
| Heater Current (Hc) | 8–20 mA |
| Coolant Flow Rate (Fc) | 0–0.7 L/min |

---

## 📈 Performance Highlights

| Metric | Result |
|----------|---------|
| Temperature Tracking Accuracy | ±2°C |
| Thermal Runaway Events | 0 |
| Resilience Score | > 0.93 |
| Batch Duration | 60 Hours |
| Control Strategy | PPO |

---

## 📁 Project Structure

```bash
ppo-reactor-controller/
│
├── train_ppo.py
├── ppo_agent.py
├── batch_reactor_env.py
├── ukf_estimator.py
├── resilience_acrylamide.py
├── ppo_reactor.pt
│
├── explainability/
│   ├── explain_shap.py
│   ├── explain_tree.py
│   ├── explain_sensitivity.py
│   └── explain_policy_surface.py
│
├── data/
│   ├── Trajectory2.csv
│   └── Opeloop_HFc_TrTj.csv
│
├── figures/
│   ├── training_curve.png
│   ├── shap_summary.png
│   ├── sensitivity_heatmap.png
│   └── resilience_curve.png
│

```

---

## 🛠️ Technologies Used

* Python
* PyTorch
* Reinforcement Learning
* PPO (Proximal Policy Optimization)
* Unscented Kalman Filter (UKF)
* SHAP
* Scikit-learn
* NumPy
* Pandas
* Matplotlib
* Explainable AI (XAI)

---

## 🔬 Research Contributions

* Developed a resilience-oriented PPO controller for chemical process safety.
* Combined Reinforcement Learning and State Estimation for partially observable environments.
* Introduced Explainable AI techniques for interpreting control policies.
* Demonstrated safe operation of a nonlinear batch reactor without thermal runaway.
* Evaluated policy robustness using resilience engineering principles.

---

## 👨‍💻 Author

**Afhaam Ali**

Machine Learning Engineer | Reinforcement Learning Researcher

Passionate about Reinforcement Learning, Explainable AI, Process Control Systems, and Intelligent Industrial Automation.
