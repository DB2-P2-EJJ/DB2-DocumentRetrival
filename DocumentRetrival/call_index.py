from DocumentRetrival.inverted_index import MailInvertedIndex

sample_body = [[12345, "Subject: stock promo mover : cwtd * * * urgent investor trading alert * * * weekly stock pick "
                + "- - china world trade corp . ( ticker : cwtd ) * * breaking news * * china world trade corp ."
                + "enters into agreement to acquire majority stake in ceo clubs china ", 1],
               [67890, "Subject: important information thu , 30 jun 2005 . subject : important information thu ,"
                + "30 jun 2005 . thank you for using our online store and for your previous order . we have updated"
                + " our online software store - now we have more latest version of programs . our", 0],
               [15578, "Subject: urgent security notification ! dear valuedpaypalmember : we recently noticed one or"
                + "more attempts to log in to your paypal account from a foreign ip address . if you recently accessed"
                + "your account while traveling in rusia , the unusual log in attempts may have been initiated by you"
                + " . however", 1]]


def get_data(query, mii, csv):
    q = mii.query(query, 15)
    body = []
    for doc in q:
        doc_id = doc[0]
        row = csv[csv["id"] == doc_id]
        # body.append([row["id"], row["Body"], row["Label"]])
        body.append(row.values[0])
    return body


def process_index(file):
    print("Indexa", file)
    mii = MailInvertedIndex(file)
    is_correct = mii.is_sorted()
    print("Indexado")
    return mii, is_correct
