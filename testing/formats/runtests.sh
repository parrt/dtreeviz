echo "Test SVG"
dot -Tsvg test_svg.dot > test_svg.svg
grep image test_svg.svg 
dot -Tsvg:cairo test_svg.dot > test_svg-cairo.svg
grep image test_svg-cairo.svg 

echo "Test PDF"
dot -Tpdf test_pdf.dot > test_pdf.pdf
dot -Tpdf:cairo test_pdf.dot > test_pdf.pdf

echo "Test PNG"
dot -Tpng test_png.dot > test_png.png
