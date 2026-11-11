& "C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe" "D:\Data_preprocessing\LifeMate-VietASR-Embedding-ZAIC\build_fjcai_ieee_2col.py"

$docxPath = "D:\Data_preprocessing\LifeMate-VietASR-Embedding-ZAIC\docs\FJCAI_2027_LifeMate_CS_ASR_EN.docx"
$docPath = "D:\Data_preprocessing\LifeMate-VietASR-Embedding-ZAIC\docs\FJCAI_2027_LifeMate_CS_ASR_EN.doc"
$pdfPath = "D:\Data_preprocessing\LifeMate-VietASR-Embedding-ZAIC\docs\FJCAI_2027_LifeMate_CS_ASR_EN.pdf"

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$d = $word.Documents.Open($docxPath)
$pages = $d.ComputeStatistics(2)
Write-Output "Word calculated page count: $pages"
$d.SaveAs2($docPath, 0)
$d.SaveAs2($pdfPath, 17)
$d.Close()
$word.Quit()
Write-Output "Successfully exported IEEE 2-column .doc and .pdf"
