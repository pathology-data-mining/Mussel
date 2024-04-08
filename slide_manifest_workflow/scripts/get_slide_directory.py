from mussel.connectors import DremioDataframeConnector

dremioDfConnector = DremioDataframeConnector(
    snakemake.config["scheme"],
    snakemake.config["hostname"],
    snakemake.config["port"],
    snakemake.params["username"],
    snakemake.params["password"],
)
df = dremioDfConnector.get_table(snakemake.config["space"], snakemake.config["table"])
df.to_csv(snakemake.output[0], index=False)
