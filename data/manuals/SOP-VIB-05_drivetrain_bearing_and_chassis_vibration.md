# SOP-VIB-05: Drivetrain Carrier Bearing, Wheel Hub & Tire Pressure Vibration Protocol

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
