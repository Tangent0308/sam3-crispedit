"""Two-column multi-target evidence: context and an independently enlarged mask cutout."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from .fonts import load_font


def grounded_sheet(source, objects, masks, *, binary_column=False):
    width,height,header=512,352,32
    sheet=Image.new('RGB',(width*(3 if binary_column else 2),(height+header)*len(objects)),(35,35,35))
    draw=ImageDraw.Draw(sheet)
    for i,(obj,mask) in enumerate(zip(objects,masks)):
        ys,xs=np.nonzero(mask)
        if not len(xs):raise ValueError('empty group target mask')
        extent=max(int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1))
        for col,fraction in enumerate((.55,.06,.06) if binary_column else (.55,.06)):
            margin=max(24 if col==0 else 4,round(extent*fraction))
            box=(max(0,int(xs.min())-margin),max(0,int(ys.min())-margin),
                 min(source.width,int(xs.max())+margin+1),min(source.height,int(ys.max())+margin+1))
            tile=source.crop(box).convert('RGB');binary=Image.fromarray(mask[box[1]:box[3],box[0]:box[2]].astype('uint8')*255)
            scale=min(width/tile.width,height/tile.height)
            size=(max(1,round(tile.width*scale)),max(1,round(tile.height*scale)))
            m=binary.resize(size,Image.Resampling.NEAREST)
            if col==0:
                tile=tile.resize(size,Image.Resampling.LANCZOS)
                arr=np.array(tile);inside=np.asarray(m)>0
                ring=(np.asarray(m.filter(ImageFilter.MaxFilter(5)))>0)&~inside
                arr[ring]=(255,220,30);tile=Image.fromarray(arr)
            elif col==2:
                tile=m.convert("RGB")
            else:
                # Composite before AND after resizing to avoid excluded pixel leakage.
                tile=Image.composite(tile,Image.new('RGB',tile.size,(100,100,100)),binary)
                tile=Image.composite(tile.resize(size,Image.Resampling.LANCZOS),Image.new('RGB',size,(100,100,100)),m)
            y=i*(height+header)
            label=('CONTEXT / OUTSIDE CONTOUR','ENLARGED SELECTED PIXELS','WHITE = SELECTED')[col]
            draw.text((col*width+8,y+5),f'U{i+1} | {label}',font=load_font(18),fill='white')
            sheet.paste(tile,(col*width+(width-size[0])//2,y+header+(height-size[1])//2))
    return sheet
