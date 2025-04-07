public String[] minFlips(String s) {
        int n = s.length();
        int flips = 0;
        StringBuilder result = new StringBuilder(s);

        for (int i = 0; i < n;) {
            if (s.charAt(i) == '1') {
                int count = 0;
                int start = i;
                while (i < n && s.charAt(i) == '1') {
                    count++;
                    i++;
                }
                if (count == 1) {
                    if (start + 1 < n && result.charAt(start + 1) == '0') {
                        result.setCharAt(start + 1, '1');
                        flips++;
                    }
                }
            }
            else {
                int count = 0;
                int start = i;
                while (i < n && s.charAt(i) == '0') {
                    count++;
                    i++;
                }
                if (count < 3) {
                    for (int j = start; j < start + (3 - count) && j < n; j++) {
                        if (result.charAt(j) == '0') {
                            result.setCharAt(j, '1');
                            flips++;
                        }
                    }
                }
            }
        }

    return new String[]{result.toString(), String.valueOf(flips)};
}
