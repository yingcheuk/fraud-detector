# Data Dictionary: Insurance Fraud Dataset

This document contains inferred descriptions of each column based on domain knowledge and data inspection.  

**Note: Column descriptions are inferred from column names, value patterns, and insurance domain knowledge. Interpretations may be refined during EDA.**  

---

## Customer & Policy Information

- `months_as_customer`: Duration the customer has held the policy (in months)  - **numerical**  
- `policy_number`: Unique policy ID  - **nominal**  
- `policy_bind_date`: Date the policy started  - **datetime**  
- `policy_state`: State where the policy was issued  - **categorial**
- `policy_csl`: Split limit — Likely represents bodily injury liability limits per person and per accident (e.g., `250/500`)  - **categorial**  
- `policy_deductable`: Deductible amount in the policy  - **numerical**  
- `policy_annual_premium`: Yearly premium paid  - **numerical**  
- `umbrella_limit`: Additional liability coverage beyond standard policy  - **numerical**  

---

## About Insured Person

- `age`: Age of the insured person  - **numerical**  
- `insured_zip`: ZIP code  - **nominal**  
- `insured_sex`: Gender  - **categorial**  
- `insured_education_level`: Education level  - **categorial**  
- `insured_occupation`: Job/industry  - **categorial**  
- `insured_hobbies`: Hobby (possibly a lifestyle risk indicator)  - **categorial**  
- `insured_relationship`: Relationship to the policyholder  - **categorial**  
- `capital-gains`: Profit from investments, reflecting the insured person's financial profile  - **numerical**  
- `capital-loss`: Loss from investments, indicating the insured person's financial performance  - **numerical**  

---

## Incident Details

- `incident_date`: Date of the incident  - **datetime**  
- `incident_type`: Type of incident (e.g., Collision, Theft)  - **categorial**  
- `collision_type`: Type of collision (e.g., Front, Side, Rear)  - **categorial**  
- `incident_severity`: Severity level (e.g., Minor, Major, Total Loss)  - **categorial**  
- `authorities_contacted`: Authority contacted (e.g., Police, Fire)  - **categorial**  
- `incident_state`: Incident location — state  - **categorial**  
- `incident_city`: Incident location — city  - **categorial**  
- `incident_location`: Address or area  - **nominal**  
- `incident_hour_of_the_day`: Hour the incident occurred (0–23)  - **numerical**  

---

## Claims & Damages

- `number_of_vehicles_involved`: Number of cars involved  - **numerical**  
- `property_damage`: Was property damaged? (Yes/No/Unknown)  - **categorial**  
- `bodily_injuries`: Number of people injured  - **numerical**  
- `witnesses`: Number of witnesses  - **numerical**  
- `police_report_available`: Is police report available (Yes/No/Unknown)  - **categorial**    
- `total_claim_amount`: Total claim in dollars  - **numerical**  
- `injury_claim`: Sub-claim for injury  - **numerical**  
- `property_claim`: Sub-claim for property  - **numerical**  
- `vehicle_claim`: Sub-claim for vehicle  - **numerical**  

---

## Vehicle Information

- `auto_make`: Car manufacturer  - **categorial**  
- `auto_model`: Car model  - **categorial**  
- `auto_year`: Car year  - **numerical**  

---

## Target

- `fraud_reported`: Whether the claim was reported as fraud (`Y`/`N`)  - **boolean**  

---