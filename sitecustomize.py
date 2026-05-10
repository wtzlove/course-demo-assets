import site


try:
    user_site = site.getusersitepackages()
    if user_site:
        site.addsitedir(user_site)
except Exception:
    pass
