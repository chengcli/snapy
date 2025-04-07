#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  char* resultString;
  int flips;
} Result;

Result minFlips(const char* s) {
  int n = strlen(s);
  int flips = 0;
  char* result = strdup(s);  // Duplicate the input string to modify

  for (int i = 0; i < n;) {
    if (s[i] == '0') {
      int count = 0;
      int start = i;
      while (i < n && s[i] == '0') {
        count++;
        i++;
      }

      if (count < 3) {
        for (int j = start; j < start + (3 - count) && j < n; j++) {
          if (result[j] == '0') {
            result[j] = '1';
            flips++;
          }
        }
      }
    } else {
      int count = 0;
      int start = i;
      while (i < n && s[i] == '1') {
        count++;
        i++;
      }

      if (count == 1) {
        if (start + 1 < n && result[start + 1] == '0') {
          result[start + 1] = '1';
          flips++;
        }
      }
    }
  }

  Result res;
  res.resultString = result;
  res.flips = flips;
  return res;
}

int main() {
  const char* input = "10100";
  Result res = minFlips(input);

  printf("Modified String: %s\n", res.resultString);
  printf("Number of Flips: %d\n", res.flips);

  free(res.resultString);  // Free the allocated memory
  return 0;
}
