# Phase 0, steps 5 and 6: verify every dataset and record provenance in SOURCES.md.
#   MaleCNS  - compares each file's MD5 with the one Janelia's storage publishes (needs internet)
#   BabySlakh - hashes the local archive against the Zenodo MD5 (a few seconds)
#   Slakh2100 redux - reads the download log for "DONE OK"; add -RehashSlakh to re-hash all
#                     48.7 GB of parts yourself (several minutes)
# Writes reports\phase0_verify.txt (one OK/FAIL line per dataset) and SOURCES.md.
#
#   powershell -ExecutionPolicy Bypass -File .\scripts\phase0\3_verify_data.ps1 [-RehashSlakh]

param(
    [switch]$RehashSlakh,
    [string]$Base = 'C:\Users\ricos\Documents\AI Databases\flybeats'
)
$ErrorActionPreference = 'Stop'
$Repo = Join-Path $Base '.01'
$malecns = Join-Path $Base 'data\malecns'
$slakh = Join-Path $Base 'data\slakh'
$lines = @()
$rows = @()

function Hex($bytes) { -join ($bytes | ForEach-Object { $_.ToString('x2') }) }
function Day($path) { (Get-Item $path).LastWriteTime.ToString('yyyy-MM-dd') }

# --- MaleCNS v1.0 -------------------------------------------------------------
$files = 'body-annotations-male-cns-v1.0-minconf-0.5.feather',
         'body-neurotransmitters-male-cns-v1.0.feather',
         'connectome-weights-male-cns-v1.0-minconf-0.5.feather'
foreach ($f in $files) {
    $obj = "v1.0/connectome-data/flat-connectome/$f"
    $api = 'https://storage.googleapis.com/storage/v1/b/flyem-male-cns/o/' + [uri]::EscapeDataString($obj)
    $meta = curl.exe -s $api | ConvertFrom-Json
    $remote = Hex ([Convert]::FromBase64String($meta.md5Hash))
    $path = Join-Path $malecns $f
    $local = (Get-FileHash $path -Algorithm MD5).Hash.ToLower()
    $size = (Get-Item $path).Length
    $ok = ($remote -eq $local) -and ([int64]$meta.size -eq $size)
    $lines += "{0}  MaleCNS    {1}" -f $(if ($ok) { 'OK  ' } else { 'FAIL' }), $f
    $rows += [pscustomobject]@{ Name = "MaleCNS $f"; Version = 'v1.0'; Url = "https://storage.googleapis.com/flyem-male-cns/$obj"
                                Bytes = $size; Md5 = $local; Date = (Day $path) }
}

# --- BabySlakh ----------------------------------------------------------------
$baby = Join-Path $slakh 'babyslakh_16k.tar.gz'
$babyMd5 = (Get-FileHash $baby -Algorithm MD5).Hash.ToLower()
$ok = $babyMd5 -eq '311096dc2bde7d61c97e930edbfc7f78'
$lines += "{0}  BabySlakh  babyslakh_16k.tar.gz" -f $(if ($ok) { 'OK  ' } else { 'FAIL' })
$rows += [pscustomobject]@{ Name = 'BabySlakh (16 kHz)'; Version = 'babyslakh_16k'; Url = 'https://zenodo.org/records/4603870'
                            Bytes = (Get-Item $baby).Length; Md5 = $babyMd5; Date = (Day $baby) }

# --- Slakh2100 redux 16 kHz ---------------------------------------------------
$expected = '66a2301ed7b4d5f4f6d3383474e546c6'
$parts = Get-ChildItem (Join-Path $slakh 'slakh2100_redux_16k.tar.gz.part*') | Where-Object { $_.Name -match '\.part\d+$' } | Sort-Object Name
$bytes = ($parts | Measure-Object Length -Sum).Sum
if ($RehashSlakh) {
    "Re-hashing $($parts.Count) Slakh parts ($([math]::Round($bytes / 1GB, 1)) GB)..."
    $md5 = [Security.Cryptography.MD5]::Create()
    $buf = New-Object byte[] (8MB)
    foreach ($p in $parts) {
        $fs = [IO.File]::OpenRead($p.FullName)
        while (($n = $fs.Read($buf, 0, $buf.Length)) -gt 0) { [void]$md5.TransformBlock($buf, 0, $n, $null, 0) }
        $fs.Close()
    }
    [void]$md5.TransformFinalBlock($buf, 0, 0)
    $ok = (Hex $md5.Hash) -eq $expected
    $how = 're-hashed'
} else {
    $log = Get-Content (Join-Path $slakh 'parallel_download.log') -Tail 3
    $ok = ($log -match "MD5 $expected matches").Count -gt 0 -and ($log -match 'DONE OK').Count -gt 0
    $how = 'download log'
}
$ok = $ok -and ($bytes -eq 48689473348)
$lines += "{0}  Slakh2100  slakh2100_redux_16k.tar.gz ({1}, {2} bytes)" -f $(if ($ok) { 'OK  ' } else { 'FAIL' }), $how, $bytes
$rows += [pscustomobject]@{ Name = 'Slakh2100 redux (16 kHz)'; Version = 'slakh2100_redux_16k'; Url = 'https://zenodo.org/records/7708270'
                            Bytes = $bytes; Md5 = $expected; Date = (Day $parts[-1].FullName) }

# --- Write results --------------------------------------------------------------
$lines
$utf8 = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllLines((Join-Path $Repo 'reports\phase0_verify.txt'), [string[]]$lines, $utf8)

$md = @('# Sources', '',
        'Checked by `scripts/phase0/3_verify_data.ps1`. Checksums are MD5, hex.', '',
        '| Dataset | Version | URL | Size (bytes) | MD5 | Downloaded | License |',
        '| --- | --- | --- | --- | --- | --- | --- |')
foreach ($r in $rows) { $md += "| $($r.Name) | $($r.Version) | $($r.Url) | $($r.Bytes) | $($r.Md5) | $($r.Date) | CC BY 4.0 |" }
$md += '', '## Credit', '',
       ('- **MaleCNS v1.0** - Janelia FlyEM male central nervous system connectome, CC BY 4.0. Berg, Beckett, Costa, Schlegel, Januszewski et al., "Sexual dimorphism in the complete connectome of the Drosophila male central nervous system", Cell (2026); preprint bioRxiv, https://doi.org/10.1101/2025.10.09.680999. Data: https://male-cns.janelia.org'),
       '- **Slakh2100 / BabySlakh** - Manilow, Wichern, Seetharaman and Le Roux, "Cutting Music Source Separation Some Slakh", WASPAA 2019. CC BY 4.0.'
[IO.File]::WriteAllLines((Join-Path $Repo 'SOURCES.md'), [string[]]$md, $utf8)
"wrote SOURCES.md and reports\phase0_verify.txt"

$failed = ($lines -match '^FAIL').Count
if ($failed) { "$failed dataset(s) FAILED."; exit 1 }
"All datasets verified. Next: review SOURCES.md and config\locked.yaml, then scripts\phase0\4_lock_and_tag.ps1"
