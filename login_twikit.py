import asyncio, getpass
from twikit import Client

async def main():
    client = Client("en-US")
    auth1 = input("X username/email (auth_info_1): ").strip()
    auth2 = input("Email/phone (optional, Enter to skip): ").strip()
    pw = getpass.getpass("Password: ")

    kwargs = {"auth_info_1": auth1, "password": pw}
    if auth2:
        kwargs["auth_info_2"] = auth2

    await client.login(**kwargs)
    client.save_cookies("cookies.json")
    print("saved cookies.json")

asyncio.run(main())
