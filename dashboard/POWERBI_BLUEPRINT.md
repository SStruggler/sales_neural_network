# Power BI Dashboard Blueprint
## Deep Learning Based Retail Revenue Forecasting & Customer Sales Analytics

---

## Data Source
**File:** `data/processed/powerbi_clean_sales.csv`
**Supplemental:** `data/processed/predictions.csv` (Actual vs Predicted from Neural Network)

---

## Data Model (Star Schema)

```
                    ┌─────────────────┐
                    │   fact_Sales    │
                    │─────────────────│
                    │ Row ID (PK)     │
                    │ Order ID        │
                    │ Customer ID     │
                    │ Product ID      │
                    │ Sales           │
                    │ Quantity        │
                    │ Discount        │
                    │ Profit          │
                    │ Profit Margin % │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
┌────────▼────────┐ ┌────────▼────────┐ ┌───────▼─────────┐
│  dim_Date       │ │  dim_Customer   │ │  dim_Product     │
│─────────────────│ │─────────────────│ │──────────────────│
│ Order Date (PK) │ │ Customer ID(PK) │ │ Product ID (PK)  │
│ Order Year      │ │ Customer Name   │ │ Product Name     │
│ Order Month     │ │ Segment         │ │ Category         │
│ Order Month Name│ │ City            │ │ Sub-Category     │
│ Order Quarter   │ │ State           │ │                  │
│ Order Week      │ │ Country         │ │                  │
│ Is Weekend      │ │ Region          │ │                  │
└─────────────────┘ └─────────────────┘ └──────────────────┘
```

---

## DAX Measures Reference

```dax
-- Core KPIs
Total Revenue      = SUM(fact_Sales[Sales])
Total Transactions = COUNTROWS(fact_Sales)
Avg Order Value    = AVERAGE(fact_Sales[Sales])
Total Profit       = SUM(fact_Sales[Profit])
Avg Discount Rate  = AVERAGE(fact_Sales[Discount])
Profit Margin %    = DIVIDE(SUM(fact_Sales[Profit]), SUM(fact_Sales[Sales])) * 100

-- MoM Growth
Revenue MoM Growth =
VAR CurrentMonth = [Total Revenue]
VAR PrevMonth = CALCULATE([Total Revenue], DATEADD(dim_Date[Order Date], -1, MONTH))
RETURN DIVIDE(CurrentMonth - PrevMonth, PrevMonth)

-- YoY Growth
Revenue YoY Growth =
CALCULATE([Total Revenue], SAMEPERIODLASTYEAR(dim_Date[Order Date]))

-- Running Total
Revenue Running Total =
CALCULATE([Total Revenue], FILTER(ALL(dim_Date), dim_Date[Order Date] <= MAX(dim_Date[Order Date])))

-- Pareto Cumulative %
Category Cumulative % =
DIVIDE(
    CALCULATE([Total Revenue], FILTER(ALL(dim_Product[Category]),
        [Total Revenue] >= CALCULATE([Total Revenue]))),
    CALCULATE([Total Revenue], ALL(dim_Product[Category]))
)

-- Prediction vs Actual (Page 5)
Prediction MAE =
AVERAGE(predictions[Abs_Error])

Forecast Accuracy % =
1 - DIVIDE(AVERAGE(predictions[Abs_Error]), AVERAGE(predictions[Actual_Sales]))
```

---

## Page 1 — Sales Overview (Executive Dashboard)

**Purpose:** High-level KPIs for leadership — snapshot of overall health.

**Visuals:**
| Visual | Type | Fields | Notes |
|--------|------|--------|-------|
| Total Revenue | KPI Card | SUM(Sales) | With MoM delta |
| Total Transactions | KPI Card | COUNT(Row ID) | With MoM delta |
| Avg Discount Rate | KPI Card | AVG(Discount) | Format as % |
| Total Profit | KPI Card | SUM(Profit) | With color conditional |
| Monthly Sales Trend | Line Chart | Date[Month] × Sales | Dual axis with Profit |
| Sales by Region | Donut Chart | Region × Sales | 4 segments |
| Top 10 Products | Bar Chart | Product Name × Sales | Horizontal, sorted desc |
| Sales Heatmap | Matrix | Month × Year × Sales | Conditional formatting |

**Filters:** Year slicer, Region slicer, Segment slicer

---

## Page 2 — Regional Performance

**Purpose:** Geographic analysis — identify top and underperforming territories.

**Visuals:**
| Visual | Type | Fields | Notes |
|--------|------|--------|-------|
| Sales Map | Filled Map | State × Sales | USA only, bubble size = revenue |
| Regional Growth | Clustered Bar | Region × Sales × Year | YoY comparison |
| Region KPI Grid | Multi-row Card | Region × Revenue, Profit, Txns | |
| State Ranking | Table | State × Sales × Profit Margin | Sorted by Revenue |
| Discount vs Profit | Scatter | Discount % × Profit × Region | Color by region |

**Slicer:** Year, Quarter

---

## Page 3 — Product Analysis

**Purpose:** 80/20 Pareto and margin analysis by product.

**Visuals:**
| Visual | Type | Fields | Notes |
|--------|------|--------|-------|
| Revenue Pareto Chart | Combo | Category/Sub-Cat × Revenue + Cumulative % | Line at 80% |
| Top 20 Products | Waterfall | Product Name × Sales | Sorted desc |
| Category Profitability | Treemap | Category × Sub-Cat × Profit | Color = margin |
| Price vs Discount | Scatter | Sales × Discount × Category | Identify sweet spot |
| Sub-Category Matrix | Matrix | Sub-Cat × Year × Revenue | Conditional formatting |

---

## Page 4 — Customer Segment Analysis

**Purpose:** Understand Consumer vs Corporate vs Home Office patterns.

**Visuals:**
| Visual | Type | Fields | Notes |
|--------|------|--------|-------|
| Segment Revenue Share | 100% Stacked Bar | Segment × Year × Sales | Shows shift over time |
| Segment KPI Cards | Cards | Revenue, Avg Order, Margin per Segment | 3 cards |
| Customer Acquisition | Line Chart | Month × New Customers | Based on first order date |
| Purchase Frequency | Histogram | Orders per Customer | Bin by 1-2, 3-5, 6-10, 10+ |
| Segment × Category Matrix | Matrix | Segment × Category × Sales | Conditional formatting |
| Discount Behavior | Box Plot | Segment × Discount % | Identify segment sensitivity |

---

## Page 5 — Predicted Sales Trends (Neural Network Forecast)

**Purpose:** Show the ML model's performance — compare prediction vs reality.

**Data Source:** `data/processed/predictions.csv`

**Visuals:**
| Visual | Type | Fields | Notes |
|--------|------|--------|-------|
| Actual vs Predicted | Line Chart | Index × Actual_Sales + Predicted_Sales | Dual lines, same axis |
| Error Distribution | Histogram | Residual | Centered at 0 = good |
| Forecast Accuracy | Gauge | `Forecast Accuracy %` | Target: ≥ 70% |
| MAE KPI | Card | `Prediction MAE` | |
| Residuals Scatter | Scatter | Predicted_Sales × Residual | Random = unbiased model |
| Error Band Table | Table | Sample rows: Actual, Predicted, Error, Error % | |

**Notes for PBI Implementation:**
1. Connect both CSV files as separate tables
2. Link via shared `Index` column if needed
3. Add dynamic title showing model name and test date

---

## Color Palette

```
Primary Cyan:    #00D4FF   (positive metrics, highlights)
Danger Red:      #FF6B6B   (losses, negative trends)
Warning Orange:  #FFA500   (alerts, caution metrics)
Success Green:   #10B981   (profit, positive deltas)
Purple Accent:   #7C5CBF   (secondary data series)
Background Dark: #0F1117
Card Background: #1A1D27
```

---

## Implementation Steps

1. Open Power BI Desktop
2. **Get Data** → Text/CSV → `data/processed/powerbi_clean_sales.csv`
3. Add second source: `data/processed/predictions.csv`
4. Open **Power Query** and verify data types (Sales, Profit = Decimal; Dates = Date)
5. Create **Date Table**: `CALENDAR(DATE(2014,1,1), DATE(2018,12,31))`
6. Build the star schema relationships in **Model View**
7. Create all DAX measures in a dedicated `_Measures` table
8. Build each page using the blueprints above
9. Apply the color theme (import custom JSON theme or apply colors manually)
10. Publish to Power BI Service for sharing
