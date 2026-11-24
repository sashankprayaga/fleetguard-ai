from __future__ import annotations

from pathlib import Path
from fleetguard.config.settings import PATHS

MANUAL_DOCUMENTS: dict[str, str] = {
    "SOP-THERM-01_cooling_and_thermal_runaway.md": """# SOP-THERM-01: Heavy-Duty Diesel Cooling Loop & Thermal Excess Intervention

- **Document ID:** SOP-THERM-01
- **Applicable Models:** Mercedes-Benz Actros 1851, Volvo FH16 750, Scania R500 V8, MAN TGX 18.510, DAF XF 480
- **SAE J1939 Fault Codes:** SPN 110 FMI 0 (Engine Coolant Temp High), SPN 1639 FMI 18 (Fan Speed Drift)
- **Primary Telemetry Triggers:** `Thermal_Excess_C > +10.0°C`, `Engine_Temperature > 95.0°C`, `Engine_Temp_Roll6_Mean` upward trend, `Engine_Temp_Delta_24h > +5.0°C`
- **Standard Workshop Allocation:** 2.5 Hours | Bay Type: Heavy Powertrain Bay

## Diagnostic Thresholds & Root-Cause Triage
When `Thermal_Excess_C` exceeds +10°C above powertrain baseline while `Actual_Load` is below 85% of `Load_Capacity`, heat rejection failure is occurring in either the primary radiator core, viscous fan clutch, or wax-pellet thermostat housing. Sustained operation above 102°C risks cylinder head gasket warping and turbocharger bearing coking.

## Step-by-Step Workshop Inspection & Repair Procedure
1. **Safety Lockout:** Allow coolant loop pressure to drop below 0.3 bar before removing the expansion cap.
2. **Pressure & Leak Test:** Attach cooling system pressure tester to expansion tank; pressurize to **1.4 bar** for 15 minutes. Pressure drop > 0.1 bar indicates external hose or EGR cooler leak.
3. **Thermostat & Bypass Valve Check:** Remove thermostat housing (torque bolts: **28 Nm** on reassembly). Verify opening begins at **83.0°C** (Heavy Haul / Refrigerated) or **84.2°C** (Tanker) and reaches full 9.5 mm lift by 95°C.
4. **Radiator Core & Viscous Fan:** Back-flush radiator fins with low-pressure compressed air (max 4 bar) to clear road grime; verify PWM viscous fan lockup above 92°C.
5. **Coolant Flush & Vacuum Fill:** Drain spent coolant, install new spin-on coolant filter, and vacuum-fill with 50/50 silicate-free OAT coolant.

## Parts Bill of Materials (BOM)
- `KIT-COOL-83C`: OEM 83°C High-Flow Thermostat Cartridge + Viton Housing Seal
- `FLT-COOL-09`: Extended-Life OAT Coolant Conditioning Filter
- `FLD-OAT-20L`: 20L Premixed Heavy-Duty OAT Coolant (-37°C protection)
""",
    "SOP-EV-BATT-02_hv_battery_thermal_management.md": """# SOP-EV-BATT-02: Electric Semi High-Voltage Battery Thermal & SoC Deficit Protocol

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
""",
    "SOP-BRAKE-03_pneumatic_disc_brake_intervention.md": """# SOP-BRAKE-03: Pneumatic Air-Disc Brake Pad & Ventilated Rotor Intervention

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
""",
    "SOP-OIL-04_lubrication_and_service_interval.md": """# SOP-OIL-04: Synthetic Engine Oil Degradation & Overdue Service Interval Protocol

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
""",
    "SOP-VIB-05_drivetrain_bearing_and_chassis_vibration.md": """# SOP-VIB-05: Drivetrain Carrier Bearing, Wheel Hub & Tire Pressure Vibration Protocol

- **Document ID:** SOP-VIB-05
- **Applicable Models:** Long-Haul Tractor Units, Flatbed Carriers, Refrigerated & Tanker Combinations
- **SAE J1939 Fault Codes:** SPN 523410 FMI 3 (Chassis Accelerometer RMS Exceeded), SPN 241 FMI 18 (Tire Pressure Deviation)
- **Primary Telemetry Triggers:** `Vibration_Levels > 4.0 mm/s`, `Vibration_Roll6_Mean` rising, `Vibration_Delta_24h > +0.8 mm/s`, `Tire_Pressure < 8.0 bar`
- **Standard Workshop Allocation:** 3.5 Hours | Bay Type: Chassis Alignment & Drivetrain Bay

## Diagnostic Thresholds & Root-Cause Triage
When rolling 24-hour vibration (`Vibration_Roll6_Mean`) climbs steadily above **3.8 mm/s** alongside elevated `Mechanical_Stress_Index`, mechanical looseness or spalling is developing in the driveshaft center carrier bearing, steer-axle hub unit, or engine mounting isolators—often compounded by under-inflated steer/drive tires (`TPI` anomaly).

## Step-by-Step Workshop Inspection & Repair Procedure
1. **Triaxial Accelerometer Isolation:** Attach magnetic piezo sensors to front wheel hubs, transmission rear housing, and center carrier bearing bracket; spin drivetrain to pinpoint harmonic order (1x wheel speed vs. 3.7x driveshaft speed).
2. **Driveshaft U-Joint & Carrier Bearing:** Check universal joint radial play (max **0.08 mm**) and rubber isolator tearing on center support bearing.
3. **Wheel Hub End-Play & Tire Inflation:** Measure hub axial end-play with dial indicator (limit **0.05 mm**). Inflate steer tires to **9.0 bar** and drive tires to **8.5 bar** cold; inspect tread for cupping.
4. **Air Suspension Ride Height:** Calibrate ECAS leveling sensors to nominal frame height.

## Parts Bill of Materials (BOM)
- `KIT-DRV-BRG90`: Heavy-Duty Driveshaft Center Carrier Bearing + Isolator Bracket
- `HUB-UNIT-HD22`: Integrated Tapered Roller Wheel Hub Assembly + Viton Seal
""",
    "SOP-HUBS-06_german_autobahn_depot_routing.md": """# SOP-HUBS-06: German Autobahn Regional Service Depot & Workshop Routing Directory

- **Document ID:** SOP-HUBS-06
- **Applicable Corridors:** A1, A2, A3, A4, A5, A6, A7, A8, A9, A24, A45, A61
- **SLA Allocation:** Standard Preventive Inspection Slot = €350.00 (vs. €4,500.00 Autobahn Breakdown & Recovery)

## Regional 24/7 FleetGuard Partner Service Hubs
Dispatchers must route trucks exceeding the `0.67` 72-hour risk threshold to the nearest corridor hub based on live GPS coordinates:
- **West / Rhine-Ruhr Sector (`A1 Ruhr -> Hamburg`, `A2 Ruhr -> Berlin`, `A3 Rhine-Ruhr -> Frankfurt`, `A45`, `A61`):**
  - *Primary Hub:* **Oberhausen-Sterkrade Heavy Truck Center** (Exit A3/A2 Kreuz Oberhausen, 6 bays, EV HV certified)
  - *Secondary Hub:* **Cologne-Niehl Commercial Depot** (Exit A1/A57, 4 bays)
- **Central / Hesse Sector (`A3`, `A4 Cologne -> Dresden`, `A5 Frankfurt -> Basel`, `A7`):**
  - *Primary Hub:* **Frankfurt-Sued / Kelsterbach Logistics Workshop** (Exit A3 Frankfurter Kreuz, 8 bays, 24/7 parts depot)
  - *Secondary Hub:* **Kassel-Lohfelden Service Hub** (Exit A7/A49, specialized brake & cooling bay)
- **North Sector (`A1`, `A7 Hamburg -> Hanover`, `A24 Hamburg -> Berlin`):**
  - *Primary Hub:* **Hamburg-Harburg Port & Autobahn Depot** (Exit A1/A7, 6 bays)
  - *Secondary Hub:* **Hannover-Langenhagen Fleet Center** (Exit A2/A7)
- **South Sector (`A6 Mannheim -> Nuremberg`, `A8 Stuttgart -> Munich`, `A9 Munich -> Berlin`):**
  - *Primary Hub:* **Stuttgart-Vaihingen Commercial Service** (Exit A8/A81, 5 bays)
  - *Secondary Hub:* **Munich-Nord / Garching Heavy Workshop** (Exit A9/A99, 6 bays, EV HV certified)
- **East Sector (`A2`, `A4`, `A9`, `A24`):**
  - *Primary Hub:* **Berlin-Brandenburg / Ludwigsfelde Hub** (Exit A10/A9, 6 bays)
  - *Secondary Hub:* **Dresden-Neustadt Truck Service** (Exit A4)

## Dispatch Escalation Rules
- **CRITICAL (`Failure_Prob_72h >= 0.80`):** Mandatory diversion at next Autohof / regional hub within 50 km; do not cross mountain grades (`A7 Kasseler Berge`, `A8 Albaufstieg`).
- **HIGH (`0.67 <= Failure_Prob_72h < 0.80`):** Pre-book €350 preventive bay slot at destination corridor hub within 24 hours.
""",
}


def generate_manual_markdown_files(output_dir: Path | None = None) -> list[Path]:
    """Write structured OEM Technical Service Bulletins as Markdown files to data/manuals/."""
    target_dir = output_dir or PATHS.manuals_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    written_paths: list[Path] = []
    for filename, content in MANUAL_DOCUMENTS.items():
        file_path = target_dir / filename
        file_path.write_text(content.strip() + "\n", encoding="utf-8")
        written_paths.append(file_path)
    return written_paths


if __name__ == "__main__":
    paths = generate_manual_markdown_files()
    print(f"Generated {len(paths)} OEM maintenance manual documents in {PATHS.manuals_dir}:")
    for p in paths:
        print(f"  - {p.name}")