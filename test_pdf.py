from reportlab.pdfgen import canvas
c = canvas.Canvas("test.pdf")
c.drawString(100, 750, "We are a Series A startup with $5M ARR.")
c.save()
