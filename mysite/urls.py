from django.urls import path
from myapp import views
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path('', views.Home, name='index'),
    path('html',views.Static,name= 'static'),
    path('action',views.Audio,name = 'audio'),
    path('chat',views.Chat,name= 'chat'),
    path('chatjob',views.ChatAction,name='chataction')
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)