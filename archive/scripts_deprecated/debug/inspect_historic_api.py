import betfairlightweight

client = betfairlightweight.APIClient(
    username="dummy",
    password="dummy",
    app_key="dummy"
)
try:
    print("Historic methods:", dir(client.historic))
except Exception as e:
    print(e)
