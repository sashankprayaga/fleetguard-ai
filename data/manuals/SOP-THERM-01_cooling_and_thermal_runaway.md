# SOP-THERM-01: Heavy-Duty Diesel Cooling Loop & Thermal Excess Intervention

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
