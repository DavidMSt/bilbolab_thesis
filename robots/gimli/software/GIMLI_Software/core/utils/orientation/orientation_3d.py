import qmt

def transform_vector_from_a_to_b_frame(vector_in_a_frame, orientation_from_b_to_a):
    return qmt.rotate(orientation_from_b_to_a, vector_in_a_frame)

