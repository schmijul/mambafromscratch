#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc > 1 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
        puts("Usage: ./bin/train [--help]");
        puts("\nTraining implementation is being added in the next steps.");
        return 0;
    }

    puts("Mamba From Scratch: scaffold ready.");
    puts("Run with --help for current CLI.");
    return 0;
}
