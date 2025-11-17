# Trading Strategy Comparison - 3 Papers

## Estructura del Proyecto

Este proyecto compara 3 estrategias de trading basadas en diferentes papers académicos:

### Estrategias

1. **Paper 1: HMM Básico** (`strategy_paper1.py`)
   - Paper: "Determining Stock Trend Using Hidden Markov Model" (Ahuja & Eksombatchai, 2012)
   - Estrategia: 3 estados (Bull/Sideways/Bear), comprar en "good states"
   - Frecuencia: Moderada
   - Característica: Simple, basado en probabilidades de estados

2. **Paper 2: Mean Reversion** (`strategy_paper2.py`)
   - Paper: "Modeling Momentum and Reversals" (Stein & Pozharny, 2018)
   - Estrategia: "Buy losers, sell winners" - reversión a fair value
   - Frecuencia: Alta (trading activo)
   - Característica: Explota desviaciones de corto plazo

3. **Paper 3: Regime Switching** (`strategy_paper3.py`)
   - Paper: "Trading Bull and Bear Markets with HMM" (Donninger, 2017)
   - Estrategia: Switch entre stocks y treasuries según régimen
   - Frecuencia: Baja (solo cambios de régimen)
   - Característica: Protección en bear markets

### Módulos Compartidos

- `signals.py`: Generador de señales (HMM, Mean Reversion, Momentum)
- `backtester.py`: Motor de backtesting
- `evaluation.py`: Cálculo de métricas de performance

### Scripts Principales

- `main.py`: Punto de entrada, ejecuta comparación
- `main_compare.py`: Lógica de comparación de las 3 estrategias

## Uso

```bash
# Ejecutar comparación de las 3 estrategias
python main.py --compare SPY

# Comparar con otro símbolo
python main.py --compare QQQ
```

## Resultados

El script genera:
- `Paper1_HMM_portfolio.csv` / `Paper1_HMM_trades.csv`
- `Paper2_MeanReversion_portfolio.csv` / `Paper2_MeanReversion_trades.csv`
- `Paper3_RegimeSwitching_portfolio.csv` / `Paper3_RegimeSwitching_trades.csv`
- `strategy_comparison.csv` (tabla comparativa)

## Requisitos

```bash
pip install numpy pandas scipy scikit-learn requests yfinance
```

## Fuentes de Datos

1. **Primaria**: Thetadata Terminal (localhost:25510)
2. **Fallback**: CBOE (VIX/VXV) + Yahoo Finance

## Notas Importantes

### Diferencias Clave Entre Estrategias

| Aspecto | Paper 1 | Paper 2 | Paper 3 |
|---------|---------|---------|---------|
| **Objetivo** | Tendencia | Reversión | Protección |
| **Frecuencia** | Media | Alta | Baja |
| **Trades/año** | ~10-20 | ~50-100 | ~5-10 |
| **Riesgo** | Medio | Alto | Bajo |
| **Complejidad** | Baja | Alta | Media |

### Parámetros Ajustables

**Paper 1:**
- `confidence_threshold`: Confianza mínima en el estado (default: 0.80)
- `position_size`: Tamaño de posición (default: 1.0)

**Paper 2:**
- `entry_threshold`: Z-score para entrar (default: -0.3)
- `exit_threshold`: Z-score para salir (default: 0.3)
- `use_momentum_filter`: Filtrar por momentum (default: True)

**Paper 3:**
- `bear_threshold`: Umbral para bear market (default: 0.90)
- `bull_threshold`: Umbral para bull market (default: 0.70)
- `use_leverage`: Usar leverage en bull markets (default: False)

## Ejemplo de Salida

```
COMPARATIVE RESULTS
================================================================================
     Strategy  Total Return  Ann. Return  Volatility  Sharpe  Max DD  Num Trades
     Buy&Hold        0.127        0.045       0.178   0.140   0.186           0
   Paper1_HMM        0.156        0.052       0.145   0.221   0.125          15
Paper2_MeanRev        0.234        0.078       0.195   0.298   0.142          87
Paper3_RegimeSw       0.198        0.065       0.132   0.341   0.089          12
```

## Próximos Pasos

1. **Optimización de Parámetros**: Grid search para cada estrategia
2. **Walk-Forward Analysis**: Validación out-of-sample
3. **Costos de Transacción**: Refinar modelo de slippage/comisión
4. **Multi-Asset**: Extender a portfolios de múltiples activos
5. **Machine Learning**: Combinar señales con ML

## Referencias

1. Ahuja, S., & Eksombatchai, C. (2012). Determining Stock Trend Using Hidden Markov Model.
2. Stein, H. J., & Pozharny, J. (2018). Modeling Momentum and Reversals.
3. Donninger, C. (2017). Trading Bull and Bear Markets with a Hidden Markov Model.