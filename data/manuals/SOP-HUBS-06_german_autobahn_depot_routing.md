# SOP-HUBS-06: German Autobahn Regional Service Depot & Workshop Routing Directory

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
