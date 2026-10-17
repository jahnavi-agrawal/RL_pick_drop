import mujoco

xml_path = "mujoco_menagerie/franka_emika_panda/panda_pick_place.xml"

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

print("\n=== BODIES AND WORLD POSITIONS ===")
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    print(f"{i:3} | {name:30} | {data.xpos[i]}")

print("\n=== GEOMETRIES AND THEIR PARENT BODIES ===")
for i in range(model.ngeom):
    geom = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_GEOM, i
    )
    body_id = model.geom_bodyid[i]
    body = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_BODY, body_id
    )

    print(
        f"{i:3} | geom={str(geom):30} | body={str(body)}"
    )

print("\n=== SITES ===")
for i in range(model.nsite):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
    print(f"{i:3} | {name:30} | {data.site_xpos[i]}")

print("\n=== ACTUATORS ===")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"{i:3} | {name}")


print("\n=== JOINTS AND INITIAL POSITIONS ===")
for i in range(model.njnt):
    name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_JOINT, i
    )
    qadr = model.jnt_qposadr[i]
    dadr = model.jnt_dofadr[i]

    print(
        f"{i:2} | {str(name):20} "
        f"| qpos={data.qpos[qadr]:.4f} "
        f"| qvel={data.qvel[dadr]:.4f}"
    )

print("\n=== KEYFRAMES ===")
print("Number of keyframes:", model.nkey)
for i in range(model.nkey):
    name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_KEY, i
    )
    print(i, name)

