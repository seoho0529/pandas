from django.db import models

# Create your models here.
class Buser(models.Model):
    buser_no = models.IntegerField(primary_key=True)
    buser_name = models.CharField(max_length=10)
    buser_loc = models.CharField(max_length=10, blank=True, null=True)
    buser_tel = models.CharField(max_length=15, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'buser'
        
class Jikwon(models.Model):
    jikwon_no = models.IntegerField(primary_key=True)
    jikwon_name = models.CharField(max_length=10)
    buser_num = models.ForeignKey(Buser, db_column='buser_num', on_delete=models.CASCADE)

    jikwon_jik = models.CharField(max_length=10, blank=True, null=True)
    jikwon_pay = models.IntegerField(blank=True, null=True)
    jikwon_ibsail = models.DateField(blank=True, null=True)
    jikwon_gen = models.CharField(max_length=4, blank=True, null=True)
    jikwon_rating = models.CharField(max_length=3, blank=True, null=True)
 #num = models.ForeignKey(RemoteTest, help_text='논리적 외래키 테스트')
    class Meta:
        managed = False
        db_table = 'jikwon'