from suds.client import Client
from termcolor import colored
import cmd, sys

class EveShell(cmd.Cmd):

    # Cerevoice
    client = Client("https://cerevoice.com/soap/soap_1_1.php?WSDL")

    # Emotions
    emotions = [colored('Angry', 'red'),
                colored("Happy", 'blue'),
                colored("Sad", 'grey', attrs=['bold']),
                colored("Neutral", 'white'),
                colored("Frustrated", 'yellow'),
                colored("Excited", 'cyan'),
                colored("Fearful", 'magenta'),
                colored("Disgusted", 'green'),
                "Other"]
    currentEmotion = 0

    # Config
    intro = '\nWelcome to EveNet interactive shell.   Type help or ? to list commands.\n'
    prompt = '(' + emotions[currentEmotion] + ') '

    # ----- basic turtle commands -----
    def do_say(self, arg):
        'Say something using current emotion.'
        speakString = type(arg)
        request = EveShell.client.service.speakExtended("59746f2101ec1", "LJpcC67e3u", "Heather", speakString, metadata=True)

        if request.resultCode == 1:
            print(request)

        else:
            print("ERROR")
            print(request)

    def do_emo(self, arg):
        'Switch emotion according to ID'
        EveShell.currentEmotion = parse(arg)[0]
    def do_angry(self, arg):
        EveShell.currentEmotion = 0
    def do_happy(self, arg):
        EveShell.currentEmotion = 1
    def do_sad(self, arg):
        EveShell.currentEmotion = 2
    def do_neutral(self, arg):
        EveShell.currentEmotion = 3
    def do_frustrated(self, arg):
        EveShell.currentEmotion = 4
    def do_excited(self, arg):
        EveShell.currentEmotion = 5
    def do_fearful(self, arg):
        EveShell.currentEmotion = 6
    def do_disgusted(self, arg):
        EveShell.currentEmotion = 7
    def do_other(self, arg):
        EveShell.currentEmotion = 8

    # ----- record and playback -----
    def postcmd(self, stop, line):
        EveShell.prompt = '(' + EveShell.emotions[EveShell.currentEmotion] + ') '
        return False
    def close(self):
        print("byebye")


def parse(arg):
    'Convert a series of zero or more numbers to an argument tuple'
    return tuple(map(int, arg.split()))

if __name__ == '__main__':






    shell = EveShell()

    for i,emo in enumerate(shell.emotions):
        print("%d: %s" % (i, emo))

    shell.cmdloop()
