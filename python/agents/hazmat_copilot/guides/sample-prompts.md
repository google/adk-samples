# Test Prompts for Hazmat Co-Pilot

Use these prompts in the `make playground` to test the multi-agent system and RAG retrieval. The system has successfully ingested approximately 90 Safety Data Sheets, including chemicals like **Hydrogen Peroxide, Nitrobenzene, Sulfur Dioxide, Nitric Acid, Diethyl Ether, Carbon Monoxide**, and many others.

## Workplace Safety Persona (Simple & Direct)
These prompts are designed to trigger the `workplace_safety_agent`. Expect responses with pictograms, clear bullet points, and action-oriented instructions.

1. **PPE Inquiry**:
   > What specific personal protective equipment (PPE) do I need to handle Hydrogen Peroxide safely?

2. **First Aid**:
   > What should I do if someone gets Nitric Acid on their skin?

3. **Safe Storage**:
   > How should I store cylinders of Carbon Monoxide to prevent hazards?

4. **Spill Handling**:
   > What are the immediate steps to take if some Diethyl Ether spills in the lab?

5. **Hazard Clarification**:
   > What are the main fire hazards associated with Diethyl Ether?

6. **Incompatible Materials**:
   > I'm storing Nitric Acid. Can I store it next to Diethyl Ether or other organic solvents?

7. **Immediate Action on Exposure**:
   > I just splashed some Nitrobenzene in my eyes. What do I do right now?

8. **PPE for Specific Task**:
   > I need to enter an area where there might be a Sulfur Dioxide leak. What kind of respiratory protection do I need?

9. **Daily Handling / Pre-use Check**:
   > What should I check before using a cylinder of Carbon Monoxide?

## Regulatory Advisor Persona (Formal & Compliant)
These prompts are designed to trigger the `regulatory_advisor_agent`. Expect responses that reference regulations, standards (OSHA, EPA, NFPA), and formal compliance language.

10. **Regulatory Compliance**:
    > What are the OSHA Permissible Exposure Limits (PEL) and regulatory requirements for handling Hydrogen Peroxide?

11. **Environmental Reporting**:
    > Do I need to report a Nitrobenzene spill to the EPA? What are the requirements?

12. **Hazard Classification**:
    > What is the NFPA 704 rating for Sulfur Dioxide, and what do the numbers mean?

13. **Health Hazards (Formal)**:
    > Can you summarize the chronic health effects and toxicological profile of Nitrobenzene according to the SDS?

14. **Transportation**:
    > What are the DOT transportation requirements and hazard classes for Nitric Acid?

15. **Waste Disposal**:
    > What are the RCRA waste codes and disposal regulations for Diethyl Ether waste?

16. **SARA Title III**:
    > Is Sulfur Dioxide subject to SARA Title III Section 302 Extremely Hazardous Substance reporting? What is the Threshold Planning Quantity (TPQ)?

17. **Toxicological Summarization**:
    > Provide a detailed summary of the carcinogenicity and mutagenicity of Nitrobenzene as per Section 11 of the SDS.

## Complex / Multi-Persona Scenarios
These prompts test the system's ability to handle complex queries that might involve both safety and regulatory aspects, or require deep retrieval.

18. **Incident Response & Reporting**:
    > We just had a 5-gallon spill of Nitrobenzene in the warehouse. A worker was exposed on their skin. What are the immediate first aid steps, and what are our reporting obligations to the EPA/local authorities?

19. **Facility Setup**:
    > We are setting up a new storage area for Nitric Acid and Hydrogen Peroxide. What are the engineering controls and storage requirements we must implement to ensure both worker safety and regulatory compliance?
