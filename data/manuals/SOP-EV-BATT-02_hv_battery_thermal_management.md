# SOP-EV-BATT-02: Electric Semi High-Voltage Battery Thermal & SoC Deficit Protocol

- **Document ID:** SOP-EV-BATT-02
- **Applicable Models:** Mercedes-Benz eActros 600, Volvo FH Electric, Tesla Semi, Scania 45R Electric
- **UDS / ISO 14229 Fault Codes:** DTC P0A80-1C (HV Pack Thermal Imbalance), DTC P1E00-4B (Chiller Valve Flow Restriction)
- **Primary Telemetry Triggers:** `Vehicle_Type == Electric Semi`, `Battery_Deficit_Pct > 10.0%`, `Battery_Status < 65.0%`, `Thermal_Excess_C > +8.0°C`
- **Standard Workshop Allocation:** 3.0 Hours | Bay Type: High-Voltage Certified EV Bay (DGUV Information 209-093)

## Diagnostic Thresholds & Root-Cause Triage
Electric Semi battery packs operate at an optimal nominal core temperature of **65.2°C** under highway load. When `Battery_Deficit_Pct` rises alongside `Thermal_Excess_C > +8°C`, glycol flow across the battery cold-plate chiller is restricted or module cell-balancing resistors are overheating during fast-charging/regenerative braking cycles.

## Step-by-Step Workshop Inspection & Repair Procedure
1. **HV Isolation (Lockout/Tagout):** Level-3 HV technician must disconnect manual service disconnect (MSD) plug and verify < 60V DC across inverter busbars.
2. **BMS Cell Telemetry Readout:** Connect UDS diagnostic interface; inspect cell-group delta-V (maximum allowable spread: **25 mV**) and module thermistor spread (max **4.5°C**).
3. **Chiller Plate & 3-Way Thermal Valve:** Command proportional refrigerant-to-glycol chiller valve through 0%–100% sweep; verify differential pressure across battery cold plate is within **0.35–0.55 bar**.
4. **Dielectric Glycol Purge:** Flush low-conductivity dielectric coolant loop and bleed air pockets via automated vacuum purge routine.

## Parts Bill of Materials (BOM)
- `KIT-EV-CHILL-65`: Proportional 3-Way HV Chiller Valve Actuator + O-Ring Set
- `FLD-DIELEC-15L`: 15L Low-Conductivity EV Battery Thermal Glycol
