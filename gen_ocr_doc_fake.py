import os

indir = r"D:\Projects\image_text_model\data\inputs\images"
out_file = r"D:\Projects\image_text_model\data\inputs\images\ocr.csv"

filelist = []
for dirfile in os.listdir(indir):
     filename = os.fsdecode(dirfile)
     curclass = filename
     subdir = os.path.join(indir, filename)

     if os.path.isdir(subdir):
        for subfile in os.listdir(subdir):
            if subfile.endswith(".jpg") or subfile.endswith(".png") or subfile.endswith(".jpeg"):
                curfilename = subfile
                curimagefile = os.path.join(subdir, subfile)
                subfile = subfile.split(" ")
                subfile = subfile[:-1]
                subfile = ";".join(subfile)
                print (curclass + "," + curfilename + "," + str(subfile))

                filelist.append(curclass + "," + curfilename + "," + str(subfile))

with open(out_file, "w") as text_file:
    text_file.write("class, filename, text\n")

    for clist in filelist:
        text_file.write(clist)
        text_file.write("\n")