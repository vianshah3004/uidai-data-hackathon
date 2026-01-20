# Aadhaar Hotspot Model Data Statistics

## Dataset Transformation
- **Raw Input Records (Pincode Level)**: 4,326,378
- **Aggregated Records (Daily District Level)**: 238,642
  *Rationale: Raw data was at Pincode level. Aggregated to Daily District level. Features computed using 30-day (1M) and 90-day (3M) sliding windows to capture monthly trends while preserving data volume.*

## Dataset Overview
- **Total Processed Records**: 238,642
- **Total Hotspots Identified**: 142,976 (59.9%)
- **Non-Hotspots**: 95,666

## Training Split Information
The data was split temporally to prevent data leakage (using past data to predict future).

| Split | Date Range | Row Count | Hotspots | % of Data |
|-------|------------|-----------|----------|-----------|
| **Training** | 2025-03-01 to 2025-10-30 | 167,937 | 142,114 | 70% |
| **Validation** | 2025-10-31 to 2025-11-30 | 851 | 461 | 15% |
| **Testing** | 2025-12-01 to 2025-12-31 | 822 | 401 | 15% |

## Feature Space
- **Total Features Used**: 71
- **Target Variable**: `is_hotspot` (Binary)

Generated on: 2026-01-19 02:11:29.119559
