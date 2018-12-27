from random import *

def random_name():
    '''
    Generate silly names for the networks so its easier to tell them apart. Really more for fun than anything.
    '''
    first = ""
    middle = ""
    last = ""
    if randint(0, 10) > 9:
        first += choice(["Mr.", "Deputy", "Doctor", "Dark Lord", "Big", "Ultra-Rare Holographic", "Captain", "The",
                         "Mister"])
    elif randint(0, 10) > 7:
        first += choice(["Grock", "Bing", "Dur", "Bung", "Scrung", "Dung", "Ding", "Blorp", "Gronk", "Bungo Dung"
                        "Dungo Bung", "Gorp", "Dipp", "Doop"])
        if randint(0, 10) > 5:
            first += choice(["olon", "ulon", "umus", "us", "ermo", "ippus", "issimo",
                             "aretta", "etta", "-gorb", "-scrunge", "-orkus", "o"])
    else:
        first += choice(["John", "Jim", "Jane", "Janet", "Bob", "Bill", "William", "Jeremiah", "Dennis", "Bartholomew",
                         "Dungson", "Barry", "Mike", "Debbie", "Drew", "Chris", "Frungus", "Alex", "Carrie", "Ronald",
                         "Joe", "Craig", "Philliam", "Billiam", "Squilliam", "Thor", "George", "Amy", "Tyler", "Larry",
                         "Nick", "Abigail", "Mackenzie", "Allie", "Vanessa"])

    while 1:
        middle = ""
        middle += choice(["Grock", "Bing", "Dur", "Bung", "Scrung", "Dung", "Ding", "Blorp", "Gronk", "Bungo-Dung"
                         "Gorp", "Dipp", "Doop", "gro-", "gra-", "gro-", "gro-", "gra-", "gro-", "Dungo-Bung",
                          "Anthony", "Jeremiah", "Gorkus", "Borkus", "Dorkus", "Florkus"])
        if middle != first:
            break

    if randint(0, 10) > 5:
        last += choice(["Grock", "Bing", "Dur", "Bung", "Scrung", "Dung", "Ding", "Blorp", "Gronk", "Bungo-Dung"
                       "Dungo-Bung", "Gorp", "Dipp", "Doop"])
        last += choice(
            ["olon", "ulon", "umus", "er Gorbus", "us", "ermo", "ippus", "us-Bingus", "ulon Prime", "issimo", "aretta",
             "etta", "us-Dingus", "-gorb", "-scrunge", "-orkus"])
    else:
        last += choice(
            ["Agadbu", "Aglakh", "Agum", "Atumph", "Azorku", "Badbu", "Bagrat", "Bagul", "Bamog", "Bar", "Bargamph",
             "Bashnag", "Bat", "Batul", "Boga", "Bogamakh", "Bogharz", "Bogla", "Boglar", "Bogrol", "Boguk",
             "Bol", "Bolak", "Borbog", "Borbul", "Bugarn", "Bulag", "Bularz", "Bulfish", "Burbug", "Burish",
             "Burol", "Buzga", "Dugul", "Dul", "Dula", "Dulob", "Dumul", "Dumulg", "Durga", "Durog", "Durug",
             "Dush", "Gar", "Gashel", "Gat", "Ghash", "Ghasharzol", "Gholfim", "Gholob", "Ghorak", "Glorzuf",
             "Gluk", "Glurkub", "Gorzog", "Grambak", "Gulfim", "Gurakh", "Gurub", "Kashug", "Khagdum", "Kharbush",
             "Kharz", "Khash", "Khashnar", "Khatub", "Khazor", "Lag", "Lagdub", "Largum", "Lazgarn", "Loghash",
             "Logob", "Logrob", "Lorga", "Lumbuk", "Lumob", "Lurkul", "Lurn", "Luzgan", "Magar", "Magrish",
             "Mar", "Marob", "Mashnar", "Mogduk", "Moghakh", "Mughol", "Mulakh", "Murgol", "Murug", "Muzgob",
             "Muzgub", "Muzgur", "Ogar", "Ogdub", "Ogdum", "Olor", "Olurba", "Orbuma", "Rimph", "Rugob",
             "Rushub", "Shadbuk", "Shagdub", "Shagdulg", "Shagrak", "Shagramph", "Shamub", "Sharbag", "Sharga",
             "Sharob", "Sharolg", "Shatub", "Shazog", "Shug", "Shugarz", "Shugham", "Snagdu", "Uftharz", "Ugruma",
             "Urgak", "Ushar", "Ushug", "Ushul", "Uzgurn", "Yagarz"])
    if middle == "gro-" or middle == "gra-":
        return first + " " + middle + last
    else:
        return first + " " + middle + " " + last

if __name__ == "__main__":
    for i in range(200):
        print(random_name())