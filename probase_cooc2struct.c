#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef double real;

typedef struct cooccur_rec {
    int word1;
    int word2;
    real val;
} CREC;


int main(void)
{
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;

    fp = fopen("probase_cooc.txt", "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);

    FILE* fid = fopen("cooc_struct","w");
    while ((read = getline(&line, &len, fp)) != -1) {
        CREC *cr = malloc(sizeof(CREC));
        char* saveptr;
        cr->word1 = atoi(strtok_r(line, " ", &saveptr));
        cr->word2 = atoi(strtok_r(NULL, " ", &saveptr));
        cr->val = atof(strtok_r(NULL, " ", &saveptr));
        fwrite(cr, sizeof(CREC), 1, fid);
        printf("%d %d %f\n", cr->word1, cr->word2, cr->val);
        free(cr);
    }

    fclose(fp);
    if (line)
        free(line);
    exit(EXIT_SUCCESS);
}
