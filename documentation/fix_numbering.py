import re

with open("DTM_Drainage_AI_Report.tex", "r", encoding="utf-8") as f:
    content = f.read()

# Replace section numbering patterns
content = re.sub(r"\\section\{1\. ([^}]+)\}", r"\\section{\1}", content)
content = re.sub(r"\\section\{2\. ([^}]+)\}", r"\\section{\1}", content)
content = re.sub(r"\\section\{3\. ([^}]+)\}", r"\\section{\1}", content)
content = re.sub(r"\\section\{4\. ([^}]+)\}", r"\\section{\1}", content)
content = re.sub(r"\\section\{5\. ([^}]+)\}", r"\\section{\1}", content)
content = re.sub(r"\\section\{6\. ([^}]+)\}", r"\\section{\1}", content)
content = re.sub(r"\\section\{7\. ([^}]+)\}", r"\\section{\1}", content)
content = re.sub(r"\\section\{8\. ([^}]+)\}", r"\\section{\1}", content)

# Replace subsection numbering patterns
content = re.sub(r"\\subsection\{1\.1 ([^}]+)\}", r"\\subsection{\1}", content)
content = re.sub(r"\\subsection\{1\.2 ([^}]+)\}", r"\\subsection{\1}", content)
content = re.sub(r"\\subsection\{1\.3 ([^}]+)\}", r"\\subsection{\1}", content)
content = re.sub(r"\\subsection\{1\.4 ([^}]+)\}", r"\\subsection{\1}", content)

content = re.sub(r"\\subsection\{2\.1 ([^}]+)\}", r"\\subsection{\1}", content)
content = re.sub(r"\\subsection\{2\.2 ([^}]+)\}", r"\\subsection{\1}", content)
content = re.sub(r"\\subsection\{2\.3 ([^}]+)\}", r"\\subsection{\1}", content)
content = re.sub(r"\\subsection\{2\.4 ([^}]+)\}", r"\\subsection{\1}", content)
content = re.sub(r"\\subsection\{2\.5 ([^}]+)\}", r"\\subsection{\1}", content)

content = re.sub(r"\\subsection\{3\.1 ([^}]+)\}", r"\\subsection{\1}", content)
content = re.sub(r"\\subsection\{3\.2 ([^}]+)\}", r"\\subsection{\1}", content)
content = re.sub(r"\\subsection\{3\.3 ([^}]+)\}", r"\\subsection{\1}", content)
content = re.sub(r"\\subsection\{3\.4 ([^}]+)\}", r"\\subsection{\1}", content)
content = re.sub(r"\\subsection\{3\.5 ([^}]+)\}", r"\\subsection{\1}", content)
content = re.sub(r"\\subsection\{3\.6 ([^}]+)\}", r"\\subsection{\1}", content)
content = re.sub(r"\\subsection\{3\.7 ([^}]+)\}", r"\\subsection{\1}", content)
content = re.sub(r"\\subsection\{3\.8 ([^}]+)\}", r"\\subsection{\1}", content)

content = re.sub(r"\\subsection\{4\.1 ([^}]+)\}", r"\\subsection{\1}", content)
content = re.sub(r"\\subsection\{4\.2 ([^}]+)\}", r"\\subsection{\1}", content)
content = re.sub(r"\\subsection\{4\.3 ([^}]+)\}", r"\\subsection{\1}", content)
content = re.sub(r"\\subsection\{4\.4 ([^}]+)\}", r"\\subsection{\1}", content)
content = re.sub(r"\\subsection\{4\.5 ([^}]+)\}", r"\\subsection{\1}", content)

content = re.sub(r"\\subsection\{5\.1 ([^}]+)\}", r"\\subsection{\1}", content)
content = re.sub(r"\\subsection\{5\.2 ([^}]+)\}", r"\\subsection{\1}", content)
content = re.sub(r"\\subsection\{5\.3 ([^}]+)\}", r"\\subsection{\1}", content)
content = re.sub(r"\\subsection\{5\.4 ([^}]+)\}", r"\\subsection{\1}", content)
content = re.sub(r"\\subsection\{5\.5 ([^}]+)\}", r"\\subsection{\1}", content)
content = re.sub(r"\\subsection\{5\.6 ([^}]+)\}", r"\\subsection{\1}", content)

content = re.sub(r"\\subsection\{6\.1 ([^}]+)\}", r"\\subsection{\1}", content)
content = re.sub(r"\\subsection\{6\.2 ([^}]+)\}", r"\\subsection{\1}", content)
content = re.sub(r"\\subsection\{6\.3 ([^}]+)\}", r"\\subsection{\1}", content)
content = re.sub(r"\\subsection\{6\.4 ([^}]+)\}", r"\\subsection{\1}", content)

content = re.sub(r"\\subsection\{7\.1 ([^}]+)\}", r"\\subsection{\1}", content)
content = re.sub(r"\\subsection\{7\.2 ([^}]+)\}", r"\\subsection{\1}", content)
content = re.sub(r"\\subsection\{7\.3 ([^}]+)\}", r"\\subsection{\1}", content)
content = re.sub(r"\\subsection\{7\.4 ([^}]+)\}", r"\\subsection{\1}", content)

content = re.sub(r"\\subsection\{8\.1 ([^}]+)\}", r"\\subsection{\1}", content)
content = re.sub(r"\\subsection\{8\.2 ([^}]+)\}", r"\\subsection{\1}", content)

with open("DTM_Drainage_AI_Report.tex", "w", encoding="utf-8") as f:
    f.write(content)

print("Done! Removed manual section numbering.")
