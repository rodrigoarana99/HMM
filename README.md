# Estrategia de Trading HMM Optimizada

Sistema de trading cuantitativo basado en **Hidden Markov Models (HMM)** para detección de régimen de mercado y **Regime Switching** para asignación de activos.

## 📊 Estrategia

Basada en el paper académico:
> **"Trading Bull and Bear Markets with a Hidden Markov Model"**
> Donninger, C. (2017)

### Lógica de Trading

El sistema detecta automáticamente el régimen de mercado usando HMM y ajusta la exposición:

| Régimen | Probabilidad de Riesgo | Asignación | Objetivo |
|---------|------------------------|------------|----------|
| 🟢 **BULL** | < 0.50 | 100% Stocks | Crecimiento |
| 🟡 **TRANSICIÓN** | 0.50 - 0.75 | 70% Stocks / 30% Cash | Cautela |
| 🔴 **BEAR** | > 0.75 | 100% Cash | Protección |

### Características Clave

✅ **Detección automática de régimen** - HMM de 2 estados (bull/bear)
✅ **Protección en bear markets** - Sale a cash cuando detecta alto riesgo
✅ **Trading moderado** - 8-12 trades/año (evita overtrading)
✅ **Anti-whipsaw** - Requiere 5 días de confirmación antes de cambiar régimen
✅ **Datos de VIX** - Usa VIX term structure para mejorar detección

## 🚀 Uso Rápido

### Instalación

```bash
pip install numpy pandas scipy scikit-learn yfinance
```

### Ejecutar Backtest

```bash
python run_optimized.py [SYMBOL] [START_DATE] [END_DATE]
```

**Ejemplos:**

```bash
# SPY desde 2020
python run_optimized.py SPY 2020-01-01 2024-01-01

# QQQ últimos 3 años (por defecto)
python run_optimized.py QQQ
```

### Resultados

El script genera:
- `optimized_portfolio.csv` - Historial diario del portfolio
- `optimized_trades.csv` - Detalle de todos los trades ejecutados

## 📁 Estructura del Proyecto

### Archivos Principales

| Archivo | Descripción |
|---------|-------------|
| `run_optimized.py` | ⭐ Script principal - ejecuta backtest completo |
| `optimized_strategy.py` | Estrategia HMM optimizada |
| `hmm_detector.py` | Implementación del HMM simplificado |
| `backtester.py` | Motor de backtesting |
| `evaluation.py` | Métricas de performance |

### Papers Académicos

```
PDFs/
├── Trading Bull- and Bear-Markets with a Hidden Markov Model.pdf  ⭐ Paper principal
├── Modeling Momentum and Reversals.pdf
└── Determining Stock Trend Using HHM.pdf
```

## 🔧 Parámetros Ajustables

En `optimized_strategy.py`:

```python
strategy = OptimizedHMMStrategy(
    bear_threshold=0.75,         # Umbral para bear market
    bull_threshold=0.50,         # Umbral para bull market
    transition_stock_pct=0.70,   # % stocks en transición
    min_regime_days=5            # Días mínimos antes de cambiar
)
```

## 📈 Métricas de Performance

El sistema calcula automáticamente:

- **Annualized Return** - Retorno anualizado
- **Sharpe Ratio** - Return ajustado por riesgo
- **Max Drawdown** - Pérdida máxima desde peak
- **Win Rate** - % de trades ganadores

## 🧠 Cómo Funciona el HMM

### Observables

El HMM usa 2 features:

1. **Absolute Returns** - Volatilidad anualizada
2. **VIX Term Structure** - VIX3M - VIX (contango vs backwardation)

### Estados

- **Estado 0 (Bull)**: Baja volatilidad + VIX en contango
- **Estado 1 (Bear)**: Alta volatilidad + VIX en backwardation

### Algoritmo

1. **Baum-Welch (EM)** - Entrena parámetros del HMM
2. **Forward Algorithm** - Calcula probabilidades de estado

## 🎯 Objetivos de Performance

| Métrica | Target |
|---------|--------|
| Sharpe Ratio | > 0.5 |
| Max Drawdown | < 20% |
| Trades/año | 8-15 |
| Ann. Return | > 8% |

## 📝 Changelog

### Versión Optimizada (2024-11-17)

- ✅ Simplificación radical del código
- ✅ Eliminadas estrategias no rentables (Papers 1 y 2)
- ✅ Enfoque único en Regime Switching (Paper 3)
- ✅ Umbrales optimizados: bear=0.75, bull=0.50
- ✅ Código limpio: ~400 líneas vs 4400 anteriores

---

**Última actualización:** 2024-11-17