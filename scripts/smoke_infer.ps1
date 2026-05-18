param(
    [string]$Python = "D:\coding\anaconda\envs\qwen\python.exe",
    [string]$Image = "data\pm_hmcw\raw\real-world\test\image_2\000248.png",
    [string]$Calib = "data\pm_hmcw\raw\real-world\test\calib\000248.txt",
    [string]$OutJson = "runs\smoke_000248.json",
    [string]$OutVis = "runs\smoke_000248.png"
)

& $Python -m human_detect.infer `
    --image $Image `
    --calib $Calib `
    --out $OutJson `
    --vis $OutVis `
    --imgsz 640 `
    --geom-size 640 `
    --num-tokens 1200 `
    --device cuda:0 `
    --half

