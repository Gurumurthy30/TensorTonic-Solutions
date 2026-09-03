def percent_change(series: list) -> list:
    out = []
    for i in range(1,len(series)):
        prev_element = series[i-1]
        crr = series[i]
        # xi - xi-1
        if(prev_element!=0):
            percent = (crr - prev_element) / prev_element
        else:
            percent = 0.0
        out.append(percent)
    return out