#!/usr/bin/env python3
"""Six portfolio PNGs built exclusively from real browser screenshots."""
import asyncio, math, os
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageStat
from playwright.async_api import async_playwright

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"portfolio"/"assets"
RAW=ROOT/"portfolio"/"captures"
OUT.mkdir(parents=True,exist_ok=True)
RAW.mkdir(parents=True,exist_ok=True)
W,H=1600,1000
SITES=[
  ("studioboiler","https://studioboiler.be/en/",(219,142,72)),
  ("wolvesart","https://wolvesart.fr/",(130,129,113)),
  ("interviewer","https://interviewer.ai/",(83,105,200)),
  ("panora","https://getpanora.com/",(25,170,105)),
  ("visily","https://www.visily.ai/",(111,89,224)),
  ("rheaply","https://rheaply.com/",(195,121,71)),
]
LAYOUTS={
 "studioboiler":[("d",[(120,190),(1145,130),(1170,745),(140,820)]),("m",[(1040,240),(1370,205),(1455,825),(1125,880)])],
 "wolvesart":[("d",[(265,100),(1400,220),(1315,780),(185,695)]),("m",[(65,300),(430,230),(475,885),(95,915)])],
 "interviewer":[("d",[(120,145),(1250,155),(1190,795),(155,810)]),("m",[(1060,205),(1390,280),(1320,900),(1005,832)])],
 "panora":[("d",[(155,130),(1230,190),(1195,805),(120,735)]),("m",[(1010,180),(1400,220),(1465,855),(1090,845)])],
 "visily":[("m",[(70,300),(345,260),(382,860),(91,895)]),("d",[(270,135),(1335,110),(1360,740),(270,800)]),("m",[(1130,370),(1400,350),(1430,905),(1160,902)])],
 "rheaply":[("d",[(130,180),(1265,105),(1320,740),(190,830)]),("m",[(1060,240),(1370,195),(1450,850),(1150,925)])],
}
def aspect_crop(im, ratio):
    # Keep exact screenshot pixels and proportions (no generated or redrawn UI).
    w,h=im.size
    if w/h > ratio:
        nw=max(1,int(h*ratio)); xx=(w-nw)//2; return im.crop((xx,0,xx+nw,h))
    nh=max(1,int(w/ratio)); return im.crop((0,0,w,min(h,nh)))

def panel(base,img,corners):
    q=np.float32(corners)
    width=max(150,round(max(np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[2]-q[3]))))
    height=max(150,round(max(np.linalg.norm(q[3]-q[0]),np.linalg.norm(q[2]-q[1]))))
    src=aspect_crop(img,width/height).resize((width,height),Image.Resampling.LANCZOS).convert("RGBA")
    # Opacity transitions gradually toward the edges; avoids sharp device silhouettes.
    xx=np.minimum(np.arange(width),np.arange(width)[::-1])
    yy=np.minimum(np.arange(height),np.arange(height)[::-1])
    fade=np.minimum(np.minimum(xx[None,:]/30,yy[:,None]/30),1.0)
    alpha=(np.clip(fade,0,1)*238).astype("uint8")
    src.putalpha(Image.fromarray(alpha,"L"))
    mat=cv2.getPerspectiveTransform(np.float32([[0,0],[width,0],[width,height],[0,height]]),q)
    warped=cv2.warpPerspective(np.array(src),mat,(W,H),flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))
    obj=Image.fromarray(warped,"RGBA")
    a=obj.getchannel("A")
    shadow=Image.new("RGBA",(W,H),(2,17,15,0))
    shadow.putalpha(a.filter(ImageFilter.GaussianBlur(30)).point(lambda v:int(v*.28)))
    base.alpha_composite(shadow,(8,13))
    base.alpha_composite(obj)

def glow(col):
    lyr=Image.new("RGBA",(W,H),(0,0,0,0))
    draw=ImageDraw.Draw(lyr)
    draw.ellipse((200,60,1450,940),fill=(*col,22))
    return lyr.filter(ImageFilter.GaussianBlur(150))

def render(name,col,d,m):
    canvas=Image.new("RGBA",(W,H),(0,0,0,0))
    canvas.alpha_composite(glow(col))
    for kind,quad in LAYOUTS[name]: panel(canvas,d if kind=="d" else m,quad)
    out=OUT/f"{name}.png"
    canvas.save(out,optimize=True)
    return out

async def capture(browser,name,url,mode):
    w,h=(1440,900) if mode=="d" else (390,844)
    ctx=await browser.new_context(
         viewport={"width":w,"height":h},
         device_scale_factor=1,
         is_mobile=(mode=="m"),has_touch=(mode=="m"),
         locale="en-US",color_scheme="light",
         reduced_motion="reduce")
    page=await ctx.new_page()
    try:
        response=await page.goto(url,wait_until="domcontentloaded",timeout=60000)
        if response and response.status>=400:raise RuntimeError(f"HTTP {response.status}")
        await page.wait_for_timeout(3000)
        try:
            await page.evaluate("""() => document.fonts.ready""")
        except Exception: pass
        # Keep only genuine visible website pixels, including any legitimate cookie banner.
        target=RAW/f"{name}-{mode}.png"
        await page.screenshot(path=str(target),animations="disabled",timeout=30000)
        im=Image.open(target).convert("RGB")
        if sum(ImageStat.Stat(im.resize((80,60))).stddev)<25:
            raise RuntimeError("Screenshot looks blank or blocked")
        return im
    finally: await ctx.close()

async def main():
    async with async_playwright() as p:
        browser=await p.chromium.launch(headless=True,args=["--no-sandbox","--disable-dev-shm-usage"])
        failed=[]
        for name,url,col in SITES:
            try:
                d=await capture(browser,name,url,"d")
                m=await capture(browser,name,url,"m")
                result=render(name,col,d,m)
                print(f"CAPTURED {name}: {result} ({result.stat().st_size} bytes)",flush=True)
            except Exception as e:
                failed.append(f"{name}: {e}")
                print(f"FAILED {name}: {e}",flush=True)
        await browser.close()
        if failed: raise SystemExit("Incomplete capture, no substituted UI: "+"; ".join(failed))

if __name__=="__main__": asyncio.run(main())
