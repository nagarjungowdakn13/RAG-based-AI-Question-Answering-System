import subprocess
import shutil
from pathlib import Path

def compile_latex():
    tex_path = Path("report/paper.tex")
    if not tex_path.exists():
        print("Error: paper.tex not found in report/")
        return
    
    # Check if pdflatex is installed
    if not shutil.which("pdflatex"):
        print("\n[LaTeX Info] pdflatex was not found on your system path. Cannot compile LaTeX to PDF automatically.")
        print("However, all LaTeX sources and style sheets are written to report/paper.tex.")
        print("You can compile it online (e.g. Overleaf) or run 'pdflatex report/paper.tex' manually.")
        return
    
    print("\nFound pdflatex! Compiling LaTeX report...")
    try:
        # Run pdflatex from project root so it can resolve figures/ paths
        cmd = ["pdflatex", "-interaction=nonstopmode", "-output-directory=report", "report/paper.tex"]
        # Run twice to resolve citations, figure numbers and labels
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        pdf_file = Path("report/paper.pdf")
        if pdf_file.exists():
            print(f"LaTeX report compiled successfully: {pdf_file.resolve()}")
        else:
            print("Warning: pdflatex completed but no PDF was generated. Check report/paper.log")
    except subprocess.CalledProcessError as e:
        print("Warning: pdflatex failed during compilation. Check report/paper.log for details.")

if __name__ == "__main__":
    compile_latex()
