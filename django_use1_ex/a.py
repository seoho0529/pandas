# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class AuthGroup(models.Model):
    name = models.CharField(unique=True, max_length=150)

    class Meta:
        managed = False
        db_table = 'auth_group'


class AuthGroupPermissions(models.Model):
    id = models.BigAutoField(primary_key=True)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)
    permission = models.ForeignKey('AuthPermission', models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_group_permissions'
        unique_together = (('group', 'permission'),)


class AuthPermission(models.Model):
    name = models.CharField(max_length=255)
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING)
    codename = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'auth_permission'
        unique_together = (('content_type', 'codename'),)


class AuthUser(models.Model):
    password = models.CharField(max_length=128)
    last_login = models.DateTimeField(blank=True, null=True)
    is_superuser = models.IntegerField()
    username = models.CharField(unique=True, max_length=150)
    first_name = models.CharField(max_length=150)
    last_name = models.CharField(max_length=150)
    email = models.CharField(max_length=254)
    is_staff = models.IntegerField()
    is_active = models.IntegerField()
    date_joined = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'auth_user'


class AuthUserGroups(models.Model):
    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_groups'
        unique_together = (('user', 'group'),)


class AuthUserUserPermissions(models.Model):
    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    permission = models.ForeignKey(AuthPermission, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_user_permissions'
        unique_together = (('user', 'permission'),)


class Board(models.Model):
    user_ide = models.CharField(max_length=20, blank=True, null=True)
    business_numb = models.CharField(max_length=20, blank=True, null=True)
    board_title = models.CharField(max_length=20)
    board_content = models.TextField(blank=True, null=True)
    board_date = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'board'


class Buser(models.Model):
    buser_no = models.IntegerField(primary_key=True)
    buser_name = models.CharField(max_length=10)
    buser_loc = models.CharField(max_length=10, blank=True, null=True)
    buser_tel = models.CharField(max_length=15, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'buser'


class Contact(models.Model):
    contact_name = models.CharField(max_length=20, blank=True, null=True)
    contact_email = models.CharField(max_length=50, blank=True, null=True)
    contact_title = models.CharField(max_length=50, blank=True, null=True)
    contact_message = models.TextField(blank=True, null=True)
    contact_date = models.DateTimeField(blank=True, null=True)
    contact_status = models.CharField(max_length=20, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'contact'


class Container(models.Model):
    cont_no = models.IntegerField(primary_key=True)
    cont_addr = models.CharField(max_length=50, blank=True, null=True)
    cont_we = models.DecimalField(max_digits=10, decimal_places=6, blank=True, null=True)
    cont_kyung = models.DecimalField(max_digits=10, decimal_places=6, blank=True, null=True)
    cont_size = models.CharField(max_length=15, blank=True, null=True)
    cont_name = models.CharField(max_length=30, blank=True, null=True)
    owner_phone = models.CharField(max_length=20, blank=True, null=True)
    cont_status = models.CharField(max_length=20, blank=True, null=True)
    cont_image = models.CharField(max_length=20, blank=True, null=True)
    owner_num = models.ForeignKey('Owner', models.DO_NOTHING, db_column='owner_num')

    class Meta:
        managed = False
        db_table = 'container'


class ContainerReviews(models.Model):
    review_no = models.AutoField(primary_key=True)
    cont_no = models.ForeignKey(Container, models.DO_NOTHING, db_column='cont_no')
    user = models.ForeignKey('User', models.DO_NOTHING)
    rating = models.IntegerField()
    review_text = models.CharField(max_length=255, blank=True, null=True)
    review_date = models.DateField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'container_reviews'


class DjangoAdminLog(models.Model):
    action_time = models.DateTimeField()
    object_id = models.TextField(blank=True, null=True)
    object_repr = models.CharField(max_length=200)
    action_flag = models.PositiveSmallIntegerField()
    change_message = models.TextField()
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING, blank=True, null=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'django_admin_log'


class DjangoContentType(models.Model):
    app_label = models.CharField(max_length=100)
    model = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'django_content_type'
        unique_together = (('app_label', 'model'),)


class DjangoMigrations(models.Model):
    id = models.BigAutoField(primary_key=True)
    app = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    applied = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_migrations'


class DjangoSession(models.Model):
    session_key = models.CharField(primary_key=True, max_length=40)
    session_data = models.TextField()
    expire_date = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_session'


class Faq(models.Model):
    faq_category = models.CharField(max_length=20, blank=True, null=True)
    faq_question = models.TextField(blank=True, null=True)
    faq_answer = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'faq'


class Gogek(models.Model):
    gogek_no = models.IntegerField(primary_key=True)
    gogek_name = models.CharField(max_length=10)
    gogek_tel = models.CharField(max_length=20, blank=True, null=True)
    gogek_jumin = models.CharField(max_length=14, blank=True, null=True)
    gogek_damsano = models.ForeignKey('Jikwon', models.DO_NOTHING, db_column='gogek_damsano', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'gogek'


class Jikwon(models.Model):
    jikwon_no = models.IntegerField(primary_key=True)
    jikwon_name = models.CharField(max_length=10)
    buser_num = models.IntegerField()
    jikwon_jik = models.CharField(max_length=10, blank=True, null=True)
    jikwon_pay = models.IntegerField(blank=True, null=True)
    jikwon_ibsail = models.DateField(blank=True, null=True)
    jikwon_gen = models.CharField(max_length=4, blank=True, null=True)
    jikwon_rating = models.CharField(max_length=3, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'jikwon'


class Owner(models.Model):
    business_num = models.CharField(primary_key=True, max_length=12)
    owner_pwd = models.CharField(max_length=20)
    owner_name = models.CharField(max_length=10)
    owner_tel = models.CharField(max_length=20)
    email = models.CharField(max_length=30)
    cont_num = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'owner'


class Rv(models.Model):
    rating = models.IntegerField(blank=True, null=True)
    content = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'rv'


class Sangdata(models.Model):
    code = models.IntegerField(primary_key=True)
    sang = models.CharField(max_length=20, blank=True, null=True)
    su = models.IntegerField(blank=True, null=True)
    dan = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'sangdata'


class User(models.Model):
    user_id = models.CharField(primary_key=True, max_length=20)
    user_pwd = models.CharField(max_length=20)
    user_name = models.CharField(max_length=10)
    user_tel = models.CharField(max_length=20, blank=True, null=True)
    user_email = models.CharField(max_length=30, blank=True, null=True)
    user_addr = models.CharField(max_length=50, blank=True, null=True)
    user_jumin = models.CharField(unique=True, max_length=14, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'user'
