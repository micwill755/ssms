# State Space Models (SSMs) Learning Repository

A comprehensive collection of State Space Model implementations, from basic concepts to modern S4 architecture.

## Folder Structure

```
ssms/
├── basic/                    # Basic SSM implementations
│   ├── simple_ssm.py        # Core SSM equations
│   ├── discretized_ssm.py   # Continuous → discrete conversion
│   ├── rnn_vs_ssm.py        # RNN vs SSM comparison
│   ├── linear_state_space_layer.py  # Dual computation modes
│   ├── ssm_dual_computation.py      # Simple dual computation demo
│   ├── ssm_language_model.py        # Basic language model
│   ├── text_utils.py               # Text processing utilities
│   ├── train_ssm.py               # Training on sine waves
│   └── train_on_book.py           # Training on real text
├── s4/                      # S4 (Structured State Spaces)
│   ├── simple_s4.py         # S4 with HiPPO initialization
│   └── s4_language_model.py # S4-based language model
├── ssm_evolution.py         # Evolution: Basic → S4 → Mamba
├── terminology_guide.py     # Correct terminology
└── requirements.txt         # Dependencies
```

## Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### 1. Learn Basic SSMs
```bash
cd basic/
python simple_ssm.py           # Core SSM math
python ssm_dual_computation.py # Recurrent vs Convolution
python rnn_vs_ssm.py           # Compare with RNNs
```

### 2. Try Language Modeling
```bash
cd basic/
python train_on_book.py        # Train on real text
```

### 3. Explore S4
```bash
cd s4/
python simple_s4.py            # S4 vs Basic SSM
python s4_language_model.py    # S4 language model
```

### 4. Understand Evolution
```bash
python ssm_evolution.py        # Basic → S4 → Mamba timeline
python terminology_guide.py    # Correct names
```

## Learning Path

### Beginner
1. `basic/simple_ssm.py` - Learn core equations
2. `basic/ssm_dual_computation.py` - Understand dual computation
3. `basic/rnn_vs_ssm.py` - Compare with RNNs

### Intermediate  
1. `basic/linear_state_space_layer.py` - Full dual implementation
2. `basic/ssm_language_model.py` - Language modeling
3. `s4/simple_s4.py` - S4 improvements

### Advanced
1. `s4/s4_language_model.py` - S4 language model
2. `ssm_evolution.py` - Historical context
3. Study Mamba papers for selective SSMs

## Key Concepts

### Basic SSMs
- **Equations**: `x(t+1) = Ax(t) + Bu(t)`, `y(t) = Cx(t)`
- **Dual Computation**: Recurrent (inference) vs Convolution (training)
- **Linear**: No nonlinearities, pure matrix operations

### S4 (Structured State Spaces)
- **HiPPO Initialization**: Special way to initialize matrix A
- **Diagonal Structure**: A is diagonal for efficiency
- **Long Sequences**: Handles 16K+ tokens without vanishing gradients

### Mamba (Selective SSMs)
- **Selective**: A,B,C matrices depend on input
- **Context-Aware**: Like attention mechanisms
- **State-of-the-Art**: Competes with Transformers

## Papers

- **HiPPO** (2020): Mathematical foundation
- **S4** (2021): "Efficiently Modeling Long Sequences with Structured State Spaces"
- **Mamba** (2023): "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
- **Mamba 2** (2024): Current state-of-the-art

## Implementation Notes

- **Pure Python**: All basic implementations use only numpy
- **Educational**: Focus on understanding over performance
- **Progressive**: Start simple, add complexity gradually
- **Dual Modes**: All models show both recurrent and convolution computation

## Fun Facts

- SSMs were invented in the 1960s for control theory
- The "dual computation" trick is what makes modern SSMs competitive
- S4's breakthrough was not new math, but better initialization
- Mamba makes SSMs "selective" like attention mechanisms

Happy learning!