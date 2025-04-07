#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Struct to hold the result string and flip count
typedef struct {
  char* resultString;
  int flips;
} Result;

Result minFlips(const char* s) {
  int n = strlen(s);
  int flips = 0;

  // Duplicate the input string to modify
  char* result = strdup(s);

  for (int i = 0; i < n;) {
    if (s[i] == '1') {
      int count = 0;
      int start = i;
      // Count consecutive '1's
      while (i < n && s[i] == '1') {
        count++;
        i++;
      }
      // If the length of consecutive '1's is less than 2, flip additional bits
      // to satisfy the condition
      if (count < 2) {
        int requiredFlips = 2 - count;
        for (int j = start + count; j < start + count + requiredFlips && j < n;
             j++) {
          if (result[j] == '0') {
            result[j] = '1';
            flips++;
          }
        }
      }
    } else {  // Case for '0's
      int count = 0;
      int start = i;
      // Count consecutive '0's
      while (i < n && s[i] == '0') {
        count++;
        i++;
      }
      // If the length of consecutive '0's is less than 3, flip additional bits
      // to satisfy the condition
      if (count < 3) {
        int requiredFlips = 3 - count;
        for (int j = start + count; j < start + count + requiredFlips && j < n;
             j++) {
          if (result[j] == '1') {
            result[j] = '0';
            flips++;
          }
        }
      }
    }
  }

  // Populate the result struct
  Result res;
  res.resultString = result;
  res.flips = flips;
  return res;
}

int main() {
  const char* input = "1101001011100010101011001";
  Result res = minFlips(input);

  printf("Modified String: %s\n", res.resultString);
  printf("Number of Flips: %d\n", res.flips);

  // Free allocated memory
  free(res.resultString);

  return 0;
}
