# Testing import of images

## Mac

### Without librsvg

Seems that importing pdf and generating pdf:

```bash
$ dot -Tpdf test_pdf.dot > test_pdf.pdf
Warning: No loadimage plugin for "pdf:cairo"
```

fails whereas importing svg and generating svg works:

```bash
$ dot -Tsvg test_svg.dot > test_svg.svg
```

<img src="test_svg.svg" width="200">

That leaves image refs in the generated .svg rather than embedding. 

W/o librsvg, using `-Tsvg:cairo` doesn't work:

```bash
$ dot -Tsvg:cairo test_svg.dot > test_svg-cairo.svg
Warning: No loadimage plugin for "svg:cairo"
```

Importing png and generating png works:

```bash
dot -Tpng test_png.dot > test_png.png
```

<img src="test_png.png" width="200">

### With librsvg

```bash
$ brew reinstall graphviz --with-librsvg --with-app --with-pango
```

then this works again:

```bash
dot -Tsvg:cairo test_svg.dot > test_svg-cairo.svg
```

<img src="test_svg-cairo.svg" width="200">

Important: this embeds the images instead of leaving refs to other files.

## Linux ubuntu 18.04

Works exactly like os x. error messages and output, though I think it might be converting import svg to png.

## Windows 10

Seems to work with 

```
dot -Tsvg test_svg.dot > test_svg_win.svg
```

I installed graphviz installer and added to PATH env variable.

Apparently `conda install python-graphviz` from "anaconda prompt" installs that and python wrapper bits.

But, couldn't view until I saw encoding error in browser.
Take out
 
```
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
```
 
and it shows up in browser!

Oh, encoding problem explains why the >= symbol is hosed on windows. Can't see the correct unicode.

Hmm..now that BOM issue with &ge; is working. Hmm...must be the conda version I installed vs raw download.
