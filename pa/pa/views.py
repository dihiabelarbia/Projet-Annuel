from django.http import HttpResponse
import datetime


def test(request):
    html = "<html><body>LEARNING EMMOTIONS</body></html>"
    return HttpResponse(html)
