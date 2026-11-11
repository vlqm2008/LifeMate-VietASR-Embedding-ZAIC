$word = New-Object -ComObject Word.Application
$word.Visible = $false
$d = $word.Documents.Open('D:\Data_preprocessing\LifeMate-VietASR-Embedding-ZAIC\docs\FJCAI_2027_LifeMate_CS_ASR_EN.docx')
$totalPages = $d.ComputeStatistics(2)
Write-Output "Total Pages: $totalPages"
$lastPage = 0
for ($i = 1; $i -le $d.Paragraphs.Count; $i++) {
    $p = $d.Paragraphs.Item($i)
    $text = $p.Range.Text.Trim()
    if ($text.Length -gt 0) {
        $page = $p.Range.Information(3)
        if ($page -ne $lastPage) {
            Write-Output "--- START OF PAGE $page ---"
            $lastPage = $page
        }
        $len = [Math]::Min(80, $text.Length)
        Write-Output "  [P$page] $($text.Substring(0, $len))"
    }
}
$d.Close()
$word.Quit()
