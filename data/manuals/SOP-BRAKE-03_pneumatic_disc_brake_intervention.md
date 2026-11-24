# SOP-BRAKE-03: Pneumatic Air-Disc Brake Pad & Ventilated Rotor Intervention

- **Document ID:** SOP-BRAKE-03
- **Applicable Models:** All 40-Ton Tractor-Trailers, Tanker Trucks, Heavy Haul & Regional Box Fleets
- **SAE J1939 Fault Codes:** SPN 1086 FMI 1 (Primary Air Pressure Low), SPN 1682 FMI 12 (Pad Wear Sensor Open/Limit)
- **Primary Telemetry Triggers:** `Brake_Condition in (Worn, Critical)`, High `UIR`, Steep Autobahn gradients (`A7 Kasseler Berge`, `A8 Albaufstieg`, `A9 Frankenwald`)
- **Standard Workshop Allocation:** 2.0 Hours | Bay Type: Axle & Safety Inspection Pit

## Diagnostic Thresholds & Root-Cause Triage
`Brake_Condition` is one of the top two global predictors of 72-hour roadside immobilization (`17.28%` SHAP share). Once friction material drops below **4.0 mm**, thermal fade increases exponentially on long Autobahn descents, risking caliper seizure or mandatory Polizei/BAG roadside prohibition.

## Step-by-Step Workshop Inspection & Repair Procedure
1. **Pad & Rotor Measurement:** Remove wheel assemblies (re-torque lug nuts to **600 Nm** in star pattern). Measure inner and outer pad thickness with digital caliper; replace entire axle set if any pad is **<= 4.0 mm**.
2. **Rotor Runout & Crack Inspection:** Minimum ventilated disc thickness is **37.0 mm** (45.0 mm new). Reject disc if radial heat cracks exceed **1.5 mm** width or extend into the friction chamfer.
3. **Caliper Guide Pin & Adjuster Check:** Verify automatic slack adjuster mechanism rotates freely with < 15 Nm torque; inspect rubber boots for tears.
4. **Pneumatic Circuit & EBS Test:** Verify dual-circuit air compressor cut-out at **12.5 bar** and run EBS modulator solenoid actuation test.

## Parts Bill of Materials (BOM)
- `KIT-BRK-HD430`: 430mm Heavy-Duty Low-Copper Brake Pad Axle Set + Pad Retainer Clips
- `SNS-BRK-W02`: Continuous Potentiometer Brake Wear Sensor Harness
- `DSC-VENT-430`: 430x45mm Ventilated Alloy Brake Rotor (if below 37.0 mm)
