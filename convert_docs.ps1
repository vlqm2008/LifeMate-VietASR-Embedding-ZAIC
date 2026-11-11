& "C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe" "D:\Data_preprocessing\LifeMate-VietASR-Embedding-ZAIC\generate_paper_doc.py"

$docxPath = "D:\Data_preprocessing\LifeMate-VietASR-Embedding-ZAIC\docs\FJCAI_2027_LifeMate_CS_ASR_EN.docx"
$docPath = "D:\Data_preprocessing\LifeMate-VietASR-Embedding-ZAIC\docs\FJCAI_2027_LifeMate_CS_ASR_EN.doc"
$pdfPath = "D:\Data_preprocessing\LifeMate-VietASR-Embedding-ZAIC\docs\FJCAI_2027_LifeMate_CS_ASR_EN.pdf"

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$doc = $word.Documents.Open($docxPath)
$pages = $doc.ComputeStatistics(2)
Write-Output "Word calculated page count: $pages"
$doc.SaveAs2($docPath, 0)
$doc.SaveAs2($pdfPath, 17)
$doc.Close()
$word.Quit()
Write-Output "Successfully exported .doc and .pdf"
