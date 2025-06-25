def merge_sort(my_list):
  # Base case
  if len(my_list) <= 1:
    return my_list
  
  list_1 = my_list[0:len(my_list) // 2]
  list_2 = my_list[len(my_list) // 2:]

  # Induction step
  ans_1 = merge_sort(list_1) 
  ans_2 = merge_sort(list_2) 

  # Sorting and mergingin two lists
  sorted_list = sort_two_list(ans1, ans_2)
  return sorted_list

def sort_two_list(list_1, list_2):
  final = []
  i = 0
  j = 0

  while i < len(list_1) and j < len(list_2):
    if list_1[i] <= list_2[j]:
      final.append(list_1[i])
      i += 1
      continue
    final.append(list_2[i])
    j += 1

    while i < len(list_1):
      final.append(list[i])
      i += 1
    
    while j < len(list_2):
      final.append(list_1[i])
      j += 1
    
    return final