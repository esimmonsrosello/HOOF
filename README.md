# HOOF: Harmonised Optimization of Oligos and Frames

**HOOF** is a professional bioinformatics application for sequence optimization and analysis. It integrates multiple optimization algorithms with comprehensive analytical tools, all accessible through an intuitive **Streamlit** interface.

---

## Features

### Optimization Methods
- **Standard Codon Optimization** – Uses the most frequent codons for each amino acid to maximize expression.
- **Balanced Optimization** – Advanced algorithm that considers codon usage alongside +1 frame effects.
- **MaxStop** – Specialized for introducing stop codons in alternative reading frames (+1 frame).

### Sequence Analysis
- **In-Frame Analysis** – Calculates CAI, GC content, and identifies +1 slippery motifs, visualized in interactive plots.
- **+1 Frame Analysis** – Maps +1 stop codon positions and evaluates potential immunogenic +1 peptides.

### Capabilities
- **Single sequence processing** with real-time validation.
- **Batch processing** for multiple sequences via FASTA or text files.
- Configurable algorithm parameters for custom optimization workflows.

---

## Installation

Follow the instructions in **instructions.txt** for setting up a Python virtual environment and installing dependencies.

### Required Files
All necessary files are included in this GitHub repository.

### Basic Workflow
1. Open the application and navigate to the **Single Sequence** tab (for individual sequences) or **Batch Processing** tab (for multiple sequences).  
2. Paste your DNA sequence or upload a file.  
3. Select an optimization method and run the analysis.  
4. View, visualize, and download your results.

---

## File Structure

```
├── HOOF.v3.minimal.py       # Main Streamlit application
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── HumanCodons.xlsx         # Codon usage data
```

## Dependencies

- pandas
- biopython
- matplotlib
- openpyxl
- xlsxwriter
- streamlit
- requests
- beautifulsoup4
- python-dotenv
- kaleido

## Support

For issues or questions, refer to the "About" tab within the application for detailed method descriptions and requirements.


