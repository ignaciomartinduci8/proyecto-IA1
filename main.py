from CLI import CLI

LBLUE = '\033[38;5;39m'
RESET = '\033[0m'


def main():
    cli = CLI()
    cli.prompt = f"{LBLUE} --> {RESET} "
    cli.cmdloop()


if __name__ == "__main__":
    main()
