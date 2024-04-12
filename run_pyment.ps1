$files = Get-ChildItem -Path "." -Recurse -Filter "*.py"

foreach ($file in $files) {
    Write-Host "$($file.FullName) in progress..."
    & pyment $file.FullName
}

Write-Host "All code files have been processed!"