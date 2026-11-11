$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir

$pythonExe = "C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

# 1. Rebuild IEEE 2-Column Word Document
& $pythonExe "$scriptDir\build_fjcai_ieee_2col.py"

$paperDir = "$projectRoot\docs\paper"
$submissionDir = "$projectRoot\submission"
if (-not (Test-Path $submissionDir)) {
    New-Item -ItemType Directory -Path $submissionDir -Force | Out-Null
}

$docxPath = "$paperDir\FJCAI_2027_LifeMate_CS_ASR_EN.docx"
$docPath = "$paperDir\FJCAI_2027_LifeMate_CS_ASR_EN.doc"
$pdfPath = "$paperDir\FJCAI_2027_LifeMate_CS_ASR_EN.pdf"

# 2. Convert using Word COM Engine (Exact pagination & print formatting)
$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {
    $d = $word.Documents.Open($docxPath)
    $pages = $d.ComputeStatistics(2)
    Write-Output "Word calculated page count: $pages"
    
    # 0 = wdFormatDocument (doc), 17 = wdFormatPDF (pdf)
    $d.SaveAs2($docPath, 0)
    $d.SaveAs2($pdfPath, 17)
    $d.Close()
    Write-Output "[OK] Successfully exported IEEE 2-column .doc and .pdf in $paperDir"
}
finally {
    $word.Quit()
}

# 3. Synchronize camera-ready copies to submission directory
Copy-Item $docxPath "$submissionDir\FJCAI_2027_LifeMate_CS_ASR_EN.docx" -Force
Copy-Item $docPath "$submissionDir\FJCAI_2027_LifeMate_CS_ASR_EN.doc" -Force
Copy-Item $pdfPath "$submissionDir\FJCAI_2027_LifeMate_CS_ASR_EN.pdf" -Force

if (Test-Path "D:\Dataset\LifeMate_CS_ASR\docs") {
    Copy-Item "$paperDir\FJCAI_2027_LifeMate_CS_ASR_EN.*" "D:\Dataset\LifeMate_CS_ASR\docs\" -Force
}
Write-Output "[OK] Synchronized camera-ready files to submission/ and external mirror."
