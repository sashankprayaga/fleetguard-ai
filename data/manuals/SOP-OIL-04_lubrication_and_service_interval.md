# SOP-OIL-04: Synthetic Engine Oil Degradation & Overdue Service Interval Protocol

- **Document ID:** SOP-OIL-04
- **Applicable Models:** All Internal Combustion Heavy Commercial Powertrains (Euro VI-E)
- **SAE J1939 Fault Codes:** SPN 100 FMI 18 (Engine Oil Pressure Moderately Low), SPN 98 FMI 17 (Oil Quality / Viscosity Limit)
- **Primary Telemetry Triggers:** `Days_Since_Last_Maintenance > 75`, `Oil_Quality < 55.0%`, `Oil_Degradation_Rate` elevated, High `Usage_Hours`
- **Standard Workshop Allocation:** 1.5 Hours | Bay Type: Express Lubrication & Service Bay

## Diagnostic Thresholds & Root-Cause Triage
`Days_Since_Last_Maintenance` (`22.17%` SHAP share) and `Oil_Quality` (`8.19%` SHAP share) together account for over 30% of model risk attribution. High-load highway duty causes shear-thinning of 5W-30/10W-40 low-SAPS synthetic oil and soot loading in the DPF regeneration cycle, accelerating main bearing and camshaft lobe wear.

## Step-by-Step Workshop Inspection & Repair Procedure
1. **Oil Sample & Viscosity Check:** Extract 100 mL warm oil sample via dipstick vacuum pump; check for fuel dilution (> 3.5%) or coolant glycol contamination.
2. **Crankcase Drain & Centrifuge Cleaning:** Drain hot sump oil (drain plug torque: **65 Nm** with new crush washer). Disassemble and clean centrifugal bypass rotor bowl.
3. **Dual-Stage Filter Replacement:** Replace full-flow spin-on oil filter and fine bypass filter; pre-fill filter housing with clean synthetic oil to prevent dry startup.
4. **Turbocharger Feed & Oil Pressure Verification:** Inspect turbocharger oil feed line for carbon coking; start engine and verify idle oil pressure >= **1.8 bar** and governed speed pressure >= **4.2 bar**.

## Parts Bill of Materials (BOM)
- `KIT-LUBE-5W30`: 36L ACEA E6/E9 Low-SAPS 5W-30 Full Synthetic Oil
- `FLT-OIL-DUAL`: Full-Flow + Centrifugal Bypass Oil Filter Service Pack
